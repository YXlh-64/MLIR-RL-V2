"""Backend execution and a deliberately checked MLIR reference translator.

The orchestration/measurement code is in execution_profiling.py. This module has no
configuration imports at module scope: workers set their environment first.
The reference plan translates the *actual SSA graph*, including output
accumulators, rather than guessing computations from benchmark filenames.
Unsupported operations raise Unsupported; they never become approximate graphs.
"""
from __future__ import annotations

import ctypes
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


class Unsupported(ValueError):
    """The supplied program cannot be represented faithfully by this adapter."""


DTYPES = {'f32': 'float32', 'f64': 'float64', 'i32': 'int32', 'i64': 'int64',
          'i1': 'bool', 'index': 'int64'}


def ir_inventory(module):
    """Describe static IR, without equating SSA values with live allocations.

    The graph includes shaped SSA producers, users and region arguments. Thus
    views/loop-carried values are visible instead of invented producer edges.
    Allocation counts are static sites, including sites inside loops; they are
    never advertised as a runtime peak or as distinct simultaneously live buffers.
    """
    operations, arguments = [], []

    def walk(op, path, loops):
        op = op.operation
        loop_like = {'scf.for', 'scf.forall', 'scf.parallel', 'scf.while',
                     'omp.parallel', 'omp.wsloop', 'omp.loop_nest', 'affine.for', 'affine.parallel'}
        active = loops + [path] if op.name in loop_like else loops
        operations.append((path, op, active))
        for ri, region in enumerate(op.regions):
            for bi, block in enumerate(region.blocks):
                for ai, value in enumerate(block.arguments):
                    arguments.append((f'{path}/r{ri}/b{bi}/arg{ai}', value))
                for oi, child in enumerate(block.operations):
                    walk(child, f'{path}/r{ri}/b{bi}/{oi}', active)

    walk(module.operation, 'module', [])
    shaped = lambda value: str(value.type).startswith(('tensor<', 'memref<'))
    owners = {value: key for key, value in arguments if shaped(value)}
    nodes = [dict(id=key, kind='region_argument', type=str(value.type))
             for key, value in arguments if shaped(value)]
    tags, counts, edges = {}, Counter(), []
    returned = {value for _, op, _ in operations if op.name == 'func.return' for value in op.operands}
    # Follow only explicit alias/view operations backwards from returned buffers.
    # Loop-carried or interprocedural escape analysis requires another analysis;
    # the reported non-returned sites deliberately make no liveness claim.
    aliases = {'memref.cast', 'memref.subview', 'memref.reinterpret_cast',
               'memref.expand_shape', 'memref.collapse_shape', 'memref.view'}
    alias_source = {v: op.operands[0] for _, op, _ in operations
                    if op.name in aliases for v in op.results if len(op.operands)}
    def storage_root(value):
        while value in alias_source:
            value = alias_source[value]
        return value

    escaping = set(returned)
    for _, op, _ in reversed(operations):
        if op.name in aliases and any(v in escaping for v in op.results):
            escaping.update(op.operands)
    tensor_values = intermediate_tensors = memref_values = intermediate_memrefs = 0
    allocations = []
    destinations = {}
    for key, op, loops in operations:
        counts[op.name] += 1
        tag = str(op.attributes['tag']).strip('"') if 'tag' in op.attributes else None
        if tag:
            tags.setdefault(tag, []).append(dict(id=key, operation=op.name, loop_scopes=loops))
        results = [v for v in op.results if shaped(v)]
        for value in results:
            owners[value] = key
            if str(value.type).startswith('tensor<'):
                tensor_values += 1
                if value not in returned and op.name not in ('arith.constant', 'tensor.cast'):
                    intermediate_tensors += 1
            else:
                memref_values += 1
                if value not in escaping and op.name != 'memref.get_global':
                    intermediate_memrefs += 1
        if results or any(shaped(v) for v in op.operands) or op.name.startswith('linalg.'):
            nodes.append(dict(id=key, operation=op.name, tag=tag, loop_scopes=loops,
                              result_types=[str(v.type) for v in results]))
        if op.name in ('memref.alloc', 'memref.alloca', 'memref.realloc'):
            allocations.append(dict(id=key, operation=op.name, in_loop=bool(loops),
                directly_returned_or_view=any(v in escaping for v in op.results),
                types=[str(v.type) for v in op.results]))
        # Bufferized blocks often write directly into caller-provided output
        # arguments: zero memref.alloc sites does NOT mean no intermediates.
        outputs = list(op.opview.outputs) if op.name.startswith('linalg.') and hasattr(op.opview, 'outputs') else []
        if op.name == 'memref.store':
            outputs = [op.operands[1]]
        elif op.name in ('vector.transfer_write', 'vector.store'):
            outputs = [op.operands[1]]
        for value in outputs:
            if str(value.type).startswith('memref<'):
                root = storage_root(value)
                destinations.setdefault(root, []).append(key)
    for key, op, _ in operations:
        for i, value in enumerate(op.operands):
            if shaped(value):
                edges.append(dict(producer=owners.get(value), consumer=key, operand_index=i,
                                  type=str(value.type)))
    llvm_heap = sum(1 for _, op, _ in operations if op.name == 'llvm.call' and
        'callee' in op.attributes and str(op.attributes['callee']).strip('@"') in
        ('malloc', 'aligned_alloc', 'calloc', 'realloc'))
    returned_roots = {storage_root(v) for v in returned if str(v.type).startswith('memref<')}
    buffers = [dict(value=str(value), owner=owners.get(value), type=str(value.type),
                    returned_or_explicit_view=value in returned_roots, writers=writers)
               for value, writers in destinations.items()]
    return dict(operation_counts=dict(sorted(counts.items())), tagged_operations=tags,
        tensor_result_values=tensor_values, intermediate_tensor_result_values=intermediate_tensors,
        memref_result_values=memref_values, non_returned_memref_result_values=intermediate_memrefs,
        allocation_sites=len(allocations), returned_or_view_allocation_sites=sum(
            a['directly_returned_or_view'] for a in allocations),
        non_returned_allocation_sites=sum(not a['directly_returned_or_view'] for a in allocations),
        allocations=allocations, llvm_heap_allocation_call_sites=llvm_heap,
        written_buffer_identities=len(buffers),
        intermediate_written_buffer_identities=sum(not b['returned_or_explicit_view'] for b in buffers),
        written_buffers=buffers,
        llvm_stack_allocation_sites=counts['llvm.alloca'],
        graph=dict(nodes=nodes, edges=edges, scope='Direct shaped SSA dependencies; region arguments are explicit nodes'),
        count_scope='Static SSA values/allocation sites/written buffer identities, not runtime live buffers; identities merge explicit memref views only, not loop/interprocedural aliases')


def tensor_type(ty):
    """Return a static shape/dtype; reject dynamic or unsupported element types."""
    match = re.fullmatch(r'(?:tensor|memref)<([0-9]+x)*(f32|f64|i32|i64|i1)>', str(ty))
    if not match:
        raise Unsupported(f'Expected a static ranked tensor/memref: {ty}')
    parts = str(ty).split('<', 1)[1][:-1].split('x')
    return [int(x) for x in parts[:-1]], DTYPES[parts[-1]]


def integers(attr):
    """Read MLIR integer-array attributes without evaluating their text."""
    text = str(attr)
    if text.startswith('dense<'):
        text = text.split('<', 1)[1].split('>', 1)[0]
    elif text.startswith('array<'):
        text = text.split(':', 1)[-1].rstrip('>')
    return [int(x) for x in re.findall(r'-?\d+', text)]


def reference_plan(code):
    """Parse one tensor entry into a JSON plan and immutable NumPy constants.

    MLIR bindings provide SSA identities, shapes and attributes. Only projected
    permutation maps (dimensions and constant zero) are accepted for generics.
    This covers the sampled blocks and GIN/GAT, while rejecting arbitrary affine
    expressions, dynamic control flow and missing resources explicitly.
    """
    import numpy as np
    from mlir.ir import Context, Module, DenseElementsAttr, FloatAttr, IntegerAttr

    constants, inputs, nodes, values = {}, [], [], {}
    # MLIR dense-resource payloads have a four-byte alignment header. Decoding
    # actual stored bytes is essential: random replacement weights are invalid.
    resources = dict(re.findall(r'([\w.]+): "0x([0-9A-Fa-f]+)"', code))
    with Context():
        module = Module.parse(code)
        functions = [op for op in module.body.operations
                     if op.operation.name == 'func.func' and len(op.regions[0].blocks)]
        if len(functions) != 1:
            raise Unsupported('Reference adapter requires one defined tensor entry')
        function = functions[0]
        for i, arg in enumerate(function.arguments):
            shape, dtype = tensor_type(arg.type)
            name = f'a{i}'
            values[arg] = name
            inputs.append(dict(name=name, shape=shape, dtype=dtype))

        def describe(op, scope):
            name = op.operation.name
            args = [scope[v] for v in op.operands]
            attrs = {a.name: a.attr for a in op.attributes}
            result = dict(op=name, args=args, attrs={})
            result['input_shapes'] = [tensor_type(v.type)[0] if str(v.type).startswith('tensor<')
                                      else [] for v in op.operands]
            for key in ('strides', 'dilations', 'permutation', 'dimensions', 'static_offsets',
                        'static_sizes', 'static_strides', 'static_low', 'static_high'):
                if key in attrs:
                    result['attrs'][key] = integers(attrs[key])
            for key in ('dim', 'dimension', 'predicate'):
                if key in attrs:
                    result['attrs'][key] = int(IntegerAttr(attrs[key]).value)
            if name == 'arith.constant':
                key = f'c{len(constants)}'
                value = attrs['value']
                if str(op.results[0].type).startswith('tensor<'):
                    shape, dtype = tensor_type(op.results[0].type)
                    resource = re.search(r'dense_resource<([^>]+)>', str(value))
                    if resource:
                        if resource[1] not in resources:
                            raise Unsupported(f'Missing actual weight resource: {resource[1]}')
                        raw = bytes.fromhex(resources[resource[1]])
                        itemsize = np.dtype(dtype).itemsize
                        if int.from_bytes(raw[:4], 'little') != itemsize or len(raw) != 4 + math.prod(shape)*itemsize:
                            raise Unsupported('Unexpected dense-resource payload layout')
                        array = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder('<'), offset=4).reshape(shape)
                    else:
                        array = np.asarray(DenseElementsAttr(value))
                    constants[key] = np.asarray(array, dtype=dtype).copy()
                else:
                    dtype = DTYPES.get(str(op.results[0].type))
                    if dtype is None:
                        raise Unsupported(f'Unsupported scalar constant: {op.results[0].type}')
                    scalar = FloatAttr(value).value if dtype.startswith('float') else IntegerAttr(value).value
                    constants[key] = np.asarray(scalar, dtype=dtype)
                result['constant'] = key
            if op.results:
                ty = str(op.results[0].type)
                if ty.startswith('tensor<'):
                    result['shape'], result['dtype'] = tensor_type(op.results[0].type)
                elif ty in DTYPES:
                    result['dtype'] = DTYPES[ty]
            if name == 'linalg.generic':
                maps = []
                for attr in attrs['indexing_maps']:
                    match = re.fullmatch(r'affine_map<\((.*?)\) -> \((.*?)\)>', str(attr))
                    if not match:
                        raise Unsupported(f'Cannot read indexing map: {attr}')
                    expressions = [s.strip() for s in match[2].split(',') if s.strip()]
                    if any(not re.fullmatch(r'd\d+|0', s) for s in expressions):
                        raise Unsupported(f'Generic indexing needs a dedicated reference: {attr}')
                    maps.append([int(s[1:]) if s.startswith('d') else None for s in expressions])
                result['maps'] = maps
                result['iterators'] = re.findall(r'<(parallel|reduction)>', str(attrs['iterator_types']))
                block = op.regions[0].blocks[0]
                local = dict(scope)
                result['region_args'] = []
                for i, arg in enumerate(block.arguments):
                    local[arg] = f'r{i}'
                    result['region_args'].append(f'r{i}')
                body = []
                for inner in block.operations:
                    if inner.operation.name == 'linalg.yield':
                        result['yield'] = [local[v] for v in inner.operands]
                        continue
                    child = describe(inner, local)
                    if len(inner.results) != 1:
                        raise Unsupported('Generic scalar instructions must have one result')
                    child['id'] = f's{len(body)}'
                    local[inner.results[0]] = child['id']
                    body.append(child)
                result['body'] = body
                result['input_shapes'] = [tensor_type(v.type)[0] for v in op.operands]
            if name == 'func.call':
                if str(attrs.get('callee')) != '@nanoTime':
                    raise Unsupported(f'Unsupported function call: {attrs.get("callee")}')
                result['op'] = 'timer'
            return result

        outputs = []
        operations = list(function.body.blocks[0].operations)
        used_values = {value for op in operations for value in op.operands}
        for op in operations:
            if op.operation.name == 'func.return':
                # Preparation always adds/retains the timed ABI. The final i64
                # is the timer, and is not a mathematical output to compare.
                outputs = [values[v] for v in list(op.operands)[:-1]]
                continue
            dead_secondary_outputs = (len(op.results) > 1 and op.operation.name == 'linalg.generic'
                                     and all(value not in used_values for value in list(op.results)[1:]))
            if len(op.results) != 1 and not dead_secondary_outputs:
                raise Unsupported(f'Reference adapter needs one result: {op.operation.name}')
            node = describe(op, values)
            if dead_secondary_outputs:
                # GAT's max/argmax reduction only consumes the maximum. Remove
                # the provably unused tensor results and their scalar slices;
                # reject live/coupled secondary outputs instead of guessing them.
                inputs_count = len(op.operands) - len(op.results)
                keep = list(range(inputs_count + 1))
                needed = {node['yield'][0]}
                body = []
                for child in reversed(node['body']):
                    if child['id'] in needed:
                        body.append(child)
                        needed.update(child['args'])
                if needed.intersection(node['region_args'][inputs_count + 1:]):
                    raise Unsupported('Live generic result depends on a secondary accumulator')
                node['body'] = list(reversed(body))
                node['yield'] = node['yield'][:1]
                node['dead_result_indices'] = list(range(1, len(op.results)))
                for key in ('args', 'maps', 'region_args', 'input_shapes'):
                    node[key] = [node[key][i] for i in keep]
            node['id'] = f'v{len(nodes)}'
            values[op.results[0]] = node['id']
            nodes.append(node)
    if not outputs:
        raise Unsupported('No tensor outputs found')
    return dict(inputs=inputs, nodes=nodes, outputs=outputs), constants


class TensorOps:
    """Small native-operation vocabulary shared by the two reference backends."""

    def __init__(self, tensorflow=False):
        self.tf = tensorflow
        self.lib = __import__('tensorflow' if tensorflow else 'torch')

    def tensor(self, value):
        return self.lib.convert_to_tensor(value) if self.tf else self.lib.from_numpy(value.copy())

    def reshape(self, x, shape):
        return self.lib.reshape(x, shape)

    def permute(self, x, axes):
        return self.lib.transpose(x, axes) if self.tf else self.lib.permute(x, axes)

    def expand(self, x, shape):
        return self.lib.broadcast_to(x, shape)

    def cast(self, x, dtype):
        return self.lib.cast(x, getattr(self.lib, dtype)) if self.tf else x.to(getattr(self.lib, dtype))

    def reduce(self, x, axes, kind):
        if not axes:
            return x
        if self.tf:
            return getattr(self.lib, 'reduce_' + kind)(x, axis=axes)
        operation = {'sum': 'sum', 'max': 'amax', 'min': 'amin'}[kind]
        return getattr(self.lib, operation)(x, dim=tuple(axes))

    def scalar(self, node, args, constants):
        """Execute a scalar-region instruction on broadcasted tensors."""
        name, lib = node['op'], self.lib
        if name == 'arith.constant':
            return constants[node['constant']]
        if name == 'timer':
            return 0
        if name in ('arith.addf', 'arith.addi'): return args[0] + args[1]
        if name in ('arith.subf', 'arith.subi'): return args[0] - args[1]
        if name in ('arith.mulf', 'arith.muli'): return args[0] * args[1]
        if name == 'arith.divf': return args[0] / args[1]
        if name == 'arith.negf': return -args[0]
        if name in ('arith.maximumf', 'arith.maxnumf', 'arith.maxsi'): return lib.maximum(*args)
        if name in ('arith.minimumf', 'arith.minnumf', 'arith.minsi'): return lib.minimum(*args)
        if name == 'arith.select': return lib.where(*args)
        if name in ('arith.truncf', 'arith.extf', 'arith.index_cast', 'arith.sitofp', 'arith.fptosi'):
            return self.cast(args[0], node['dtype'])
        if name in ('arith.cmpf', 'arith.cmpi'):
            p = node['attrs']['predicate']
            comparison = ({0: 'eq', 1: 'ne', 2: 'lt', 3: 'le', 4: 'gt', 5: 'ge'} if name == 'arith.cmpi'
                          else {1: 'eq', 2: 'gt', 3: 'ge', 4: 'lt', 5: 'le', 6: 'ne',
                                8: 'eq', 9: 'gt', 10: 'ge', 11: 'lt', 12: 'le', 13: 'ne'}).get(p)
            if comparison is None:
                raise Unsupported(f'Comparison predicate not implemented: {name}/{p}')
            fn = {'eq': 'equal', 'ne': 'not_equal', 'lt': 'less', 'le': 'less_equal',
                  'gt': 'greater', 'ge': 'greater_equal'}[comparison] if self.tf else comparison
            result = getattr(lib, fn)(*args)
            if name == 'arith.cmpf':
                nan = (lib.math.is_nan(args[0]) | lib.math.is_nan(args[1])) if self.tf else (
                    lib.isnan(args[0]) | lib.isnan(args[1]))
                result = (result | nan) if p >= 8 else (result & ~nan)
            return result
        if name.startswith('math.'):
            operation = name.split('.')[1]
            functions = lib.math if self.tf else lib
            return getattr(functions, operation)(*args)
        raise Unsupported(f'No native reference for {name}')

    def generic(self, node, operands, global_values, constants):
        maps, shapes = node['maps'], node['input_shapes']
        rank = len(node['iterators'])
        bounds = [1] * rank
        for mapping, shape in zip(maps, shapes):
            for dim, size in zip(mapping, shape):
                if dim is not None:
                    bounds[dim] = max(bounds[dim], size)

        def align(x, mapping, shape):
            # Constant-zero indexing selects the first slice, then projected
            # dimensions are permuted and singleton broadcast axes inserted.
            slices = tuple(0 if d is None else slice(None) for d in mapping)
            x = x[slices] if any(d is None for d in mapping) else x
            dims = [d for d in mapping if d is not None]
            if len(set(dims)) != len(dims):
                raise Unsupported('Diagonal generic indexing requires a dedicated reference')
            order = sorted(range(len(dims)), key=lambda i: dims[i])
            if order != list(range(len(dims))):
                x = self.permute(x, order)
            target = [1] * rank
            for dim, size in zip(mapping, shape):
                if dim is not None: target[dim] = size
            current_shape = [size for dim, size in zip(mapping, shape) if dim is not None]
            current_shape = [current_shape[i] for i in order]
            return self.reshape(x, target) if current_shape != target else x

        values = dict(global_values)
        values.update(zip(node['region_args'], [align(x, m, s) for x, m, s in zip(operands, maps, shapes)]))
        reductions = [i for i, kind in enumerate(node['iterators']) if kind == 'reduction']
        if len(node['yield']) != 1:
            raise Unsupported('Multi-output generic needs a dedicated reference')
        yielded, body = node['yield'][0], node['body']
        combine, term = None, None
        if reductions:
            root = next((n for n in body if n['id'] == yielded), None)
            accumulator = node['region_args'][-1]
            if root is None or accumulator not in root['args'] or len(root['args']) != 2:
                raise Unsupported('Generic reduction must end in an associative accumulator update')
            combine = {'arith.addf': 'sum', 'arith.addi': 'sum', 'arith.maximumf': 'max',
                       'arith.maxnumf': 'max', 'arith.minimumf': 'min'}.get(root['op'])
            if combine is None:
                raise Unsupported(f'Unsupported reduction: {root["op"]}')
            term = root['args'][1] if root['args'][0] == accumulator else root['args'][0]
        for instruction in body:
            if reductions and instruction['id'] == yielded:
                continue
            if instruction['op'] == 'linalg.index':
                dim = instruction['attrs']['dim']
                value = (self.lib.range(bounds[dim], dtype=self.lib.int64) if self.tf
                         else self.lib.arange(bounds[dim], dtype=self.lib.int64))
                shape = [1] * rank
                shape[dim] = bounds[dim]
                values[instruction['id']] = self.reshape(value, shape)
            else:
                values[instruction['id']] = self.scalar(instruction,
                    [values[a] for a in instruction['args']], constants)
        result = self.expand(values[term if reductions else yielded], bounds)
        remaining = [i for i in range(rank) if i not in reductions]
        if reductions:
            result = self.reduce(result, reductions, combine)
        output_dims = [d for d in maps[-1] if d is not None]
        if sorted(output_dims) != remaining:
            raise Unsupported('Generic output map must cover all parallel dimensions')
        permutation = [remaining.index(d) for d in output_dims]
        if permutation != list(range(len(remaining))):
            result = self.permute(result, permutation)
        result = self.reshape(result, node['shape'])
        if combine == 'sum': result = result + operands[-1]
        if combine == 'max': result = self.lib.maximum(result, operands[-1])
        if combine == 'min': result = self.lib.minimum(result, operands[-1])
        return result

    def convolution(self, node, args, pooling=False):
        """Use native framework kernels; preserve MLIR accumulator semantics."""
        lib, kind = self.lib, node['op']
        stride = node['attrs'].get('strides', [1, 1])
        dilation = node['attrs'].get('dilations', [1, 1])
        stride = stride * 2 if len(stride) == 1 else stride
        dilation = dilation * 2 if len(dilation) == 1 else dilation
        if len(stride) != 2 or len(dilation) != 2:
            raise Unsupported('Only 2D convolution/pooling reference kernels are supported')
        if pooling:
            kernel = node['input_shapes'][1]
            if self.tf:
                x = self.permute(args[0], [0, 2, 3, 1])
                operation = 'MAX' if kind.endswith('_max') else 'AVG'
                if kernel == [1, 1]:
                    # A one-element window is exact subsampling. tf.nn.pool
                    # rejects stride > window, although these MLIR cases are valid.
                    y = x[:, ::stride[0], ::stride[1], :]
                else:
                    y = lib.nn.pool(x, kernel, operation, strides=stride, dilations=dilation, padding='VALID')
                    if operation == 'AVG': y = y * math.prod(kernel)
                y = self.permute(y, [0, 3, 1, 2])
            else:
                import torch.nn.functional as F
                if kind.endswith('_max'):
                    y = F.max_pool2d(args[0], kernel, stride, dilation=dilation)
                elif dilation == [1, 1]:
                    y = F.avg_pool2d(args[0], kernel, stride, divisor_override=1)
                else:
                    raise Unsupported('Dilated sum pooling needs a dedicated Torch reference')
            return lib.maximum(y, args[2]) if kind.endswith('_max') else y + args[2]
        depthwise = kind == 'linalg.depthwise_conv_2d_nchw_chw'
        if kind not in ('linalg.conv_2d_nchw_fchw', 'linalg.depthwise_conv_2d_nchw_chw'):
            raise Unsupported(f'Convolution layout needs a dedicated reference: {kind}')
        if self.tf:
            x = self.permute(args[0], [0, 2, 3, 1])
            if depthwise:
                shape = node['input_shapes'][1]
                w = self.reshape(self.permute(args[1], [1, 2, 0]), shape[1:] + [shape[0], 1])
                y = lib.nn.depthwise_conv2d(x, w, [1] + stride + [1], 'VALID', dilations=dilation)
            else:
                w = self.permute(args[1], [2, 3, 1, 0])
                y = lib.nn.conv2d(x, w, [1] + stride + [1], 'VALID', dilations=[1] + dilation + [1])
            y = self.permute(y, [0, 3, 1, 2])
        else:
            import torch.nn.functional as F
            shape = node['input_shapes'][1]
            w = self.reshape(args[1], [shape[0], 1] + shape[1:]) if depthwise else args[1]
            y = F.conv2d(args[0], w, stride=stride, dilation=dilation,
                         groups=node['input_shapes'][0][1] if depthwise else 1)
        return y + args[2]

    def execute(self, node, args, values, constants):
        name, lib = node['op'], self.lib
        if name == 'linalg.generic': return self.generic(node, args, values, constants)
        if name in ('linalg.matmul', 'linalg.batch_matmul'): return lib.matmul(args[0], args[1]) + args[2]
        if name == 'linalg.add': return args[0] + args[1]
        if name == 'linalg.fill': return self.expand(args[0], node['shape'])
        if name == 'linalg.transpose': return self.permute(args[0], node['attrs']['permutation'])
        if name == 'linalg.broadcast':
            shape = list(node['input_shapes'][0])
            for dim in sorted(node['attrs']['dimensions']): shape.insert(dim, 1)
            return self.expand(self.reshape(args[0], shape), node['shape'])
        if name.startswith(('linalg.conv_', 'linalg.depthwise_conv_')): return self.convolution(node, args)
        if name.startswith('linalg.pooling_'): return self.convolution(node, args, pooling=True)
        if name == 'tensor.empty': return lib.zeros(node['shape'], dtype=getattr(lib, node['dtype']))
        if name in ('tensor.expand_shape', 'tensor.collapse_shape', 'tensor.cast'):
            return self.reshape(args[0], node['shape'])
        if name == 'tensor.concat':
            axis = node['attrs']['dim']
            return lib.concat(args, axis=axis) if self.tf else lib.cat(args, dim=axis)
        if name == 'tensor.extract_slice':
            attrs = node['attrs']
            slices = tuple(slice(o, o + n*s, s) for o, n, s in zip(
                attrs['static_offsets'], attrs['static_sizes'], attrs['static_strides']))
            return self.reshape(args[0][slices], node['shape'])
        return self.scalar(node, args, constants)


class FrameworkRunner:
    """Use one saved reference plan and the worker's CPU budget for each backend.

    Inductor/XLA compilation is lazy: creating the wrapper below is not the
    full compilation. The separate first invocation precedes warmup/timing.
    """

    def __init__(self, config, method, out, progress):
        import numpy as np
        self.method, self.out, self.current = method, out, None
        self.ops = TensorOps(tensorflow=method == 'tensorflow_xla')
        lib = self.ops.lib
        if self.ops.tf:
            lib.config.set_visible_devices([], 'GPU')
            lib.config.threading.set_intra_op_parallelism_threads(config['threads'])
            lib.config.threading.set_inter_op_parallelism_threads(1)
            lib.config.experimental.set_synchronous_execution(True)
        else:
            lib.set_num_threads(config['threads'])
            if lib.get_num_interop_threads() != 1:
                lib.set_num_interop_threads(1)
            lib.set_grad_enabled(False)
            lib.set_float32_matmul_precision('highest')
        self.version = lib.__version__
        plan = json.loads((Path(config['run']) / 'reference_plan.json').read_text())
        with np.load(Path(config['run']) / 'constants.npz', allow_pickle=False) as archive:
            constants = {k: self.ops.tensor(archive[k]) for k in archive.files}
        self.names = [a['name'] for a in config['inputs']]

        def graph(*arrays):
            values = dict(zip(self.names, arrays))
            for node in plan['nodes']:
                values[node['id']] = self.ops.execute(node, [values[n] for n in node['args']], values, constants)
            return tuple(values[name] for name in plan['outputs'])

        self.reference_source = None
        if not self.ops.tf:
            with progress.stage('reference_graph_construction_ns'):
                # FX executes the plan interpreter once, producing a plain
                # native-op forward function. Neither eager nor Inductor walks
                # Python node dictionaries/regions inside the measured call.
                # Only our generated a0..aN identifiers enter this signature;
                # CSV/MLIR text is never executed as Python source.
                names = ', '.join(self.names)
                namespace = {'graph': graph}
                exec(f'def reference({names}):\n    return graph({names})\n', namespace)
                graph = lib.fx.symbolic_trace(namespace['reference'])
                graph.graph.eliminate_dead_code()
                graph.recompile()
                self.reference_source = graph.code
        with progress.stage('backend_wrapper_ns'):
            if method == 'torch_inductor':
                self.fn = lib.compile(graph, backend='inductor', fullgraph=True, dynamic=False)
            elif method == 'tensorflow_xla':
                self.fn = lib.function(graph, jit_compile=True, autograph=False)
            else:
                self.fn = graph

    def reset(self, arrays):
        if not hasattr(self, 'inputs') or self.ops.tf:
            # TF tensors are immutable. Conversion is outside the measured call.
            self.inputs = [self.ops.tensor(a) for a in arrays]
        else:
            for dest, src in zip(self.inputs, arrays): dest.copy_(self.ops.tensor(src))

    def call(self):
        self.current = self.fn(*self.inputs)
        return self.current

    def snapshot(self):
        return [(x.numpy() if self.ops.tf else x.detach().cpu().numpy()).copy() for x in self.current]

    def release(self, value):
        self.current = None

    def artifacts(self):
        evidence = {'backend_version': self.version, 'scope': 'Static compiler evidence; not dynamic SIMD utilization'}
        if self.reference_source is not None:
            (self.out / 'reference_graph.py').write_text(self.reference_source)
            evidence['reference_graph'] = 'reference_graph.py'
        if self.ops.tf:
            try:
                (self.out / 'optimized_hlo.txt').write_text(self.fn.experimental_get_compiler_ir(*self.inputs)(stage='optimized_hlo'))
                evidence['optimized_hlo'] = 'optimized_hlo.txt'
            except Exception as error:
                evidence['hlo_error'] = str(error)
        elif self.method == 'torch_inductor':
            evidence['cache'] = str(self.out / 'inductor_cache')
            sources = list((self.out / 'inductor_cache').rglob('*.cpp'))
            evidence['generated_cpp_files'] = [str(p.relative_to(self.out)) for p in sources]
        return evidence


def initialize_mlir_openmp(cpus, threads):
    """Initialize LLVM's runtime against the worker's original CPU allocation.

    Importing V5 imports Torch/libgomp, which can bind the calling thread to its
    first OMP_PLACE. LLVM/libomp otherwise mistakes that one-CPU mask for the
    entire allocation and discards the other places. Restore only our assigned
    CPUs, then initialize and validate the exact runtime used by the MLIR JIT.
    The initial thread being bound to one place *after* this is normal.
    """
    before = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus)
    libraries = os.environ.get('MLIR_SHARED_LIBS', '').split(',')
    path = next((p for p in libraries if Path(p).name.startswith(('libomp.', 'libiomp5.'))), None)
    if not path:
        raise RuntimeError('MLIR_SHARED_LIBS must include the LLVM OpenMP runtime')
    runtime = ctypes.CDLL(path)
    runtime.omp_set_dynamic(0)
    runtime.omp_set_num_threads(threads)
    places = []
    for i in range(runtime.omp_get_num_places()):
        ids = (ctypes.c_int * runtime.omp_get_place_num_procs(i))()
        runtime.omp_get_place_proc_ids(i, ids)
        places.append(list(ids))
    if places != [[cpu] for cpu in cpus] or runtime.omp_get_max_threads() != threads:
        raise RuntimeError(f'Invalid LLVM OpenMP placement: requested {cpus}, discovered {places}')
    return dict(runtime=str(Path(path).resolve()), affinity_before_restore=before,
        assigned_cpus=list(cpus), places=places, max_threads=threads)


class MLIRRunner:
    """Compile V5 IR with explicit input/output ownership and LLVM -O3.

    Baseline and RL use exactly the same lowering pipeline. Execution profiling
    schedules once per benchmark; compilation profiling schedules per trial.
    This runner consumes the saved IR and never runs the policy itself. Parsing,
    bufferization/lowering and engine creation happen here; symbol lookup on
    first invocation can materialize remaining LLVM/JIT code.
    Inputs are restored before every invocation to prevent accumulation drift.
    """

    def __init__(self, config, method, out, progress):
        import numpy as np
        from mlir.ir import Context, Module, IntegerType
        from mlir.dialects import func, arith, linalg
        from mlir.passmanager import PassManager
        from mlir.execution_engine import ExecutionEngine
        from mlir.runtime import make_nd_memref_descriptor, as_ctype
        from rl_autoschedular_v5.transforms import transform_bufferize_and_lower_v
        from rl_autoschedular_v5.execution import _EXEC_PASS_PIPELINE

        with progress.stage('openmp_initialization_ns'):
            progress.result['openmp_placement'] = initialize_mlir_openmp(config['cpus'], config['threads'])
        self.out, self.np, self.current = out, np, None
        path = Path(config.get('ir_path') or (Path(config['run']) /
            ('methods/mlir_rl/scheduled.mlir' if method == 'mlir_rl' else 'prepared.mlir')))
        analysis = {}
        def capture(stage):
            # Inspection and artifact writes are outside the measured compiler stages.
            analysis[stage] = ir_inventory(self.module)
            (out / 'ir_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
        self.context = Context()
        with self.context:
            with progress.stage('mlir_parse_ns'):
                self.module = Module.parse(path.read_text())
            capture('input')
            with progress.stage('bufferization_ns'):
                transform_bufferize_and_lower_v(self.module)
            capture('bufferized')
            (out / 'bufferized.mlir').write_text(str(self.module))
            main = next(op for op in self.module.body.operations
                        if isinstance(op, func.FuncOp) and op.name.value == 'main')
            if len(main.type.inputs) != len(config['inputs']):
                raise Unsupported('Scheduling/bufferization changed the input ABI')
            for ty, spec in zip(main.type.inputs, config['inputs']):
                if tensor_type(ty) != (spec['shape'], spec['dtype']):
                    raise Unsupported(f'Input ABI differs from saved arrays: {spec["name"]}')
            results = list(main.type.results)
            if not isinstance(results[-1], IntegerType) or results[-1].width != 64:
                raise Unsupported('MLIR entry must end with an i64 timer result')
            self.fields = []
            for i, ty in enumerate(results[:-1]):
                shape, dtype = tensor_type(ty)
                self.fields.append((f'out_{i}', make_nd_memref_descriptor(len(shape), as_ctype(np.dtype(dtype)))))
            self.output_type = type('Outputs', (ctypes.Structure,), {'_fields_': self.fields + [('delta', ctypes.c_int64)]})
            with progress.stage('mlir_lowering_ns'):
                PassManager.parse(_EXEC_PASS_PIPELINE, self.context).run(self.module.operation)
            capture('lowered')
            (out / 'lowered.mlir').write_text(str(self.module))
            (out / 'pipeline.json').write_text(json.dumps({'pass_pipeline': _EXEC_PASS_PIPELINE,
                'llvm_opt_level': 3, 'timing_cache': False}, indent=2))
            with progress.stage('jit_engine_creation_ns'):
                self.engine = ExecutionEngine(self.module, opt_level=3, enable_object_dump=True,
                    shared_libs=[p for p in os.environ.get('MLIR_SHARED_LIBS', '').split(',') if p])
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None

    def reset(self, arrays):
        from mlir.runtime import get_ranked_memref_descriptor
        if not hasattr(self, 'inputs'):
            self.inputs = [self.np.array(a, copy=True, order='C') for a in arrays]
            self.outputs = self.output_type()
            self.args = [ctypes.pointer(ctypes.pointer(self.outputs))]
            self.args += [ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a))) for a in self.inputs]
        else:
            for target, source in zip(self.inputs, arrays): self.np.copyto(target, source)

    def call(self):
        self.engine.invoke('main', *self.args)
        return self.outputs

    def snapshot(self):
        from mlir.runtime import ranked_memref_to_numpy
        return [ranked_memref_to_numpy([getattr(self.outputs, name)]).copy() for name, _ in self.fields]

    def release(self, value):
        # Bufferized results can alias inputs or one another. Free each owned
        # allocation once, and never free memory borrowed from NumPy inputs.
        borrowed = [(a.ctypes.data, a.ctypes.data + max(1, a.nbytes)) for a in self.inputs]
        seen = set()
        for name, descriptor in self.fields:
            address = ctypes.cast(getattr(self.outputs, name).allocated, ctypes.c_void_p).value
            if address and address not in seen and not any(lo <= address < hi for lo, hi in borrowed):
                self.libc.free(address)
                seen.add(address)
            setattr(self.outputs, name, descriptor())

    def artifacts(self):
        evidence = {'scope': 'Actual JIT object/disassembly; static evidence, not dynamic utilization'}
        analysis = json.loads((self.out / 'ir_analysis.json').read_text())
        evidence['ir_analysis'] = 'ir_analysis.json'
        evidence['static_ir_counts'] = {stage: {key: value for key, value in snapshot.items()
            if isinstance(value, int)} for stage, snapshot in analysis.items()}
        try:
            self.engine.dump_to_object_file(str(self.out / 'jit.o'))
            tool = shutil.which('objdump')
            if tool:
                result = subprocess.run([tool, '-d', str(self.out / 'jit.o')], capture_output=True, text=True, timeout=60)
                (self.out / 'jit.asm').write_text(result.stdout + result.stderr)
                evidence['simd_register_mentions'] = len(re.findall(r'\b[xyz]mm\d+\b', result.stdout))
        except Exception as error:
            evidence['object_error'] = str(error)
        text = (self.out / 'lowered.mlir').read_text()
        evidence.update(mlir_vector_ops=len(re.findall(r'\bvector\.', text)),
                        openmp_parallel_regions=text.count('omp.parallel'))
        return evidence
