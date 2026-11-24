#!/usr/bin/env python3
"""Dependency experiments: do the schedules of the ops in a block, and blocks
themselves, depend on each other?

  D1 interaction   inside one block: is the whole schedule worth more or less
                   than the sum of its per-op parts?  (needs `prepare` + timed runs)
  D2 context       does the policy schedule the SAME model op the same way when
                   it appears in different (overlapping) blocks?  (`prepare --schedules-only`)
  D3 composition   block X = Y + Z (Y feeds Z): does X's time/speedup follow from
                   Y's and Z's?  (uses normal timed runs, no extra preparation)
  D4 cache         same blocks timed with warm vs cold caches (run_blocks --cache cold)

Subcommands
  prepare      [MLIR env] run the policy (greedy, no timing) and write schedule variants:
                 full_lenient   whole schedule, failed transforms skipped like v5 Env
                 only_<op>      only one op's schedule (with the producers it fused)
                 no_fusion      fusion actions replaced by plain tiled parallelization
               --schedules-only writes only the decisions (cheap; for D2)
  context      D2 analysis of prepared schedules              -> context.csv / context.json
  interaction  D1 analysis of timed ablation runs             -> interaction.csv
  composition  D3 analysis of all_results.csv                 -> composition.csv
  cache        D4 analysis of all_results.csv                 -> cache.csv

Only `prepare` imports MLIR/torch/v5. The analyses need numpy only.
"""
import argparse
import collections
import copy
import csv
import importlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import block_catalog as cat          # noqa: E402
import stats                         # noqa: E402

DEFAULT_CATALOG = HERE / 'results' / 'catalog'
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor')


# --------------------------------------------------------------------------- prepare (MLIR env)

def load_env():
    """Source .env exactly like experiment.run_experiment does, and set import paths."""
    env_file = REPO / '.env'
    if env_file.is_file():
        loaded = subprocess.run(['bash', '-c', 'set -ea; source "$1" >&2; env -0', 'bash', str(env_file)],
                                check=True, capture_output=True)
        for entry in loaded.stdout.split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)
    sys.path[:0] = [str(REPO), str(REPO / 'rl_autoschedular')]


class Policy:
    """Greedy v5 policy, set up the same way as experiment.policy_schedule()."""

    def __init__(self, checkpoint, config, workdir, impl='rl_autoschedular_v5'):
        workdir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(Path(config).read_text())
        cfg.update(dask_node_count=0, results_dir=str(workdir / 'policy_logs'))
        (workdir / 'policy_config.json').write_text(json.dumps(cfg, indent=2))
        os.environ['CONFIG_FILE_PATH'] = str(workdir / 'policy_config.json')
        os.environ['DASK_NODES'] = '0'
        os.environ['DASK_TRAINING_ONLY'] = '1'
        import torch
        self.torch = torch
        self.mod = lambda name: importlib.import_module(f'{impl}.{name}')
        self.model = self.mod('model').HiearchyModel().cpu().eval()
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt,
                                   strict=True)

    def schedule(self, name, code, max_steps=100000):
        code = self.mod('full_model').ensure_timed_main_abi(code)
        t = time.perf_counter_ns()
        features = self.mod('state').extract_bench_features_from_code(name, code, 1)
        env = self.mod('env').Env()
        state = env.reset([features], 0)
        decisions, steps = [], 0
        with self.torch.inference_mode():
            while True:
                obs = self.mod('observation').Observation.from_state(state)
                indices, _, _ = self.model.sample(obs, greedy=True)
                action = self.mod('actions').ActionSpace.action_by_index(indices[0].cpu(), state)
                decisions.append({'operation': state.operation_tag, 'action': str(action), 'ready': action.ready})
                state = env.step(state, action)
                steps += 1
                if steps > max_steps:
                    raise RuntimeError('Policy exceeded max_steps')
                if state.terminal:
                    nxt = env.get_next_op_state(state)
                    if nxt is None:
                        break
                    state = nxt
        return dict(code=features.code, history=state.transformation_history, decisions=decisions,
                    policy_ns=time.perf_counter_ns() - t)


def apply_history(code, history, keep=None, strict=False):
    """Apply per-op action lists in the order v5 does (reversed history).

    strict=True  raise on the first failing transform (experiment 01 behaviour)
    strict=False skip the rest of that op's actions and continue (v5 Env behaviour)
    """
    from mlir.ir import Context, Module
    failures, applied = [], []
    with Context():
        module = Module.parse(code)
        for op_seq in reversed(history):
            if not op_seq:
                continue
            tag = op_seq[0].operation_tag
            if keep is not None and not keep(tag):
                continue
            for action in op_seq:
                try:
                    action.apply(module)
                except Exception as error:
                    if strict:
                        raise
                    failures.append(dict(operation=tag, action=str(action), error=str(error)[:300]))
                    break
            applied.append(tag)
        return str(module), failures, applied


def fusion_groups(history):
    """op tag -> scheduling group. A producer fused by TPF joins its consumer's group."""
    group = {}
    for op_seq in reversed(history):                 # scheduling order: consumers first
        if not op_seq:
            continue
        tag = op_seq[0].operation_tag
        group.setdefault(tag, tag)
        for action in op_seq:
            if getattr(action, 'symbol', None) == 'TPF':
                group[action.new_producer_tag] = group[tag]
    return group


def without_fusion(history, policy):
    """Best effort: TPF -> TP with the same parameters; fused-producer actions
    retargeted to the original producer tag. Returns (history, note)."""
    tp_cls = policy.mod('actions.tiled_parallelization').TiledParallelization
    remap, new_history, fusions = {}, [], 0
    for op_seq in history:
        new_seq = []
        for action in op_seq:
            if getattr(action, 'symbol', None) == 'TPF':
                fusions += 1
                new_seq.append(tp_cls(list(action.parameters), operation_tag=action.operation_tag))
                remap[action.new_producer_tag] = action.producer_tag
            else:
                new_seq.append(action)
        new_history.append(new_seq)
    if not fusions:
        return None, 'no fusion actions in this schedule'
    for op_seq in new_history:
        for k, action in enumerate(op_seq):
            if action.operation_tag in remap:
                moved = copy.copy(action)
                moved.operation_tag = remap[action.operation_tag]
                if isinstance(getattr(moved, 'extras', None), dict):
                    moved.extras = dict(moved.extras, operation_tag=moved.operation_tag)
                op_seq[k] = moved
    return new_history, f'{fusions} fusion action(s) replaced'


def linalg_tags(code):
    return [cat.parse_op(m, lines, {}, i)['tag'] for i, (m, lines) in enumerate(cat.read_main(code)['ops'])]


def tag_map(original_code, features_code):
    """features tag -> original block tag (positional; AstDumper may re-tag)."""
    try:
        orig, feat = linalg_tags(original_code), linalg_tags(features_code)
    except Exception:
        return {}
    return dict(zip(feat, orig)) if len(orig) == len(feat) else {t: t for t in feat if t in orig}


def original_tag(tag, mapping):
    """Map a features tag, including fused names like 'opA_opB', back to the block tag."""
    if tag in mapping:
        return mapping[tag]
    for k in sorted(mapping, key=len, reverse=True):
        if tag.startswith(k + '_'):
            return mapping[k]
    return tag


def safe(name):
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)


def prepare_block(policy, path, out, schedules_only):
    out.mkdir(parents=True, exist_ok=True)
    original = Path(path).read_text()
    s = policy.schedule(Path(path).stem, original)
    mapping = tag_map(original, s['code'])
    groups = fusion_groups(s['history'])
    per_group = collections.defaultdict(list)
    for op_seq in reversed(s['history']):
        if op_seq:
            per_group[groups.get(op_seq[0].operation_tag, op_seq[0].operation_tag)] += [str(a) for a in op_seq]
    record = dict(block=Path(path).stem, source=str(path), policy_ns=s['policy_ns'],
                  decisions=s['decisions'], tag_map=mapping,
                  groups={original_tag(g, mapping): actions for g, actions in per_group.items()},
                  variants={})
    if not schedules_only:
        try:
            apply_history(s['code'], s['history'], strict=True)
            record['strict_ok'] = True
        except Exception as error:
            record['strict_ok'], record['strict_error'] = False, str(error)[:300]
        variants = {'full_lenient': None}
        active = [g for g, seq in per_group.items()
                  if any(not a.startswith('NT') for a in seq)]
        for g in active:
            variants[f'only_{safe(original_tag(g, mapping))}'] = g
        for name, g in variants.items():
            keep = None if g is None else (lambda tag, g=g: groups.get(tag, tag) == g)
            code, failures, applied = apply_history(s['code'], s['history'], keep=keep)
            (out / f'{name}.mlir').write_text(code)
            record['variants'][name] = dict(file=str(out / f'{name}.mlir'), failures=failures,
                                            group=None if g is None else original_tag(g, mapping))
        try:
            history, note = without_fusion(s['history'], policy)
            if history is not None:
                code, failures, _ = apply_history(s['code'], history)
                (out / 'no_fusion.mlir').write_text(code)
                record['variants']['no_fusion'] = dict(file=str(out / 'no_fusion.mlir'),
                                                       failures=failures, note=note)
            else:
                record['no_fusion_note'] = note
        except Exception as error:
            record['no_fusion_note'] = f'unavailable: {type(error).__name__}: {error}'
    (out / 'variants.json').write_text(json.dumps(record, indent=1))
    return record


def context_selection(index_path, catalog_path, n_ops, per_op, seed):
    """D2 sample: model ops that appear in >= 3 eligible blocks; up to per_op blocks each."""
    catalog = {b['name']: b for b in json.loads(Path(catalog_path).read_text())['blocks']}
    index = json.loads(Path(index_path).read_text())
    conflicts = {(c['model'], c['tag']) for c in index['tag_conflicts']}
    by_op = collections.defaultdict(list)
    for name, rel in index['blocks'].items():
        if catalog.get(name, {}).get('rl_precheck') != 'ok':
            continue
        for tag in rel['tags']:
            if (rel['model'], tag) not in conflicts:
                by_op[(rel['model'], tag)].append(name)
    candidates = sorted(k for k, v in by_op.items() if len(v) >= 3)
    rng = random.Random(f'{seed}:context')
    chosen_ops = rng.sample(candidates, min(n_ops, len(candidates)))
    blocks = set()
    for key in chosen_ops:
        names = sorted(by_op[key])
        blocks.update(random.Random(f'{seed}:{key}').sample(names, min(per_op, len(names))))
    return sorted(blocks), chosen_ops


def cmd_prepare(args):
    load_env()
    if args.selection:
        selection = json.loads(Path(args.selection).read_text())
        names = [b['name'] for b in selection['blocks'] if b.get('translation') == 'ok']
        if args.ablation_only:
            names = [b['name'] for b in selection['blocks'] if b.get('ablation')]
    elif args.context_ops:
        names, ops = context_selection(args.catalog / 'blocks_index.json', args.catalog / 'blocks_catalog.json',
                                       args.context_ops, args.context_blocks_per_op, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'context_selection.json').write_text(json.dumps(dict(ops=ops, blocks=names), indent=1))
    else:
        names = args.blocks or []
    if not names:
        raise SystemExit('No blocks: pass --selection, --context-ops or --blocks')
    policy = Policy(args.checkpoint, args.config, args.out / '_policy')
    for k, name in enumerate(names, 1):
        dest = args.out / name
        if (dest / 'variants.json').exists() and not args.rerun:
            continue
        try:
            prepare_block(policy, args.data_dir / f'{name}.mlir', dest, args.schedules_only)
            print(f'[{k}/{len(names)}] {name}: ok', flush=True)
        except Exception as error:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'variants.json').write_text(json.dumps(dict(block=name, error=f'{type(error).__name__}: {error}')))
            print(f'[{k}/{len(names)}] {name}: {type(error).__name__}: {error}', flush=True)


# --------------------------------------------------------------------------- D2 context

def cmd_context(args):
    rows, per_op = [], collections.defaultdict(dict)
    for path in sorted(args.variants.glob('*/variants.json')):
        rec = json.loads(path.read_text())
        if 'groups' not in rec:
            continue
        m = cat.BLOCK_NAME_RE.match(rec['block'])
        model = m['model'] if m else None
        mapping = rec.get('tag_map', {})
        for tag, actions in rec['groups'].items():
            # Normalise producer tags inside TPF(...) strings to block tags.
            text = ' '.join(actions)
            for feat, orig in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
                text = text.replace(feat, orig)
            per_op[(model, tag)][rec['block']] = text
    agreements = []
    for (model, tag), schedules in sorted(per_op.items()):
        if len(schedules) < 2:
            continue
        values = list(schedules.values())
        pairs = list(itertools.combinations(values, 2))
        agree = sum(a == b for a, b in pairs) / len(pairs)
        counts = collections.Counter(values)
        agreements.append(agree)
        rows.append(dict(model=model, tag=tag, blocks=len(values), distinct_schedules=len(counts),
                         modal_share=round(counts.most_common(1)[0][1] / len(values), 4),
                         pairwise_agreement=round(agree, 4),
                         blocks_list=';'.join(sorted(schedules))))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'context.csv', rows)
    summary = dict(ops_compared=len(rows),
                   mean_pairwise_agreement=stats.mean(agreements),
                   ops_always_same_schedule=sum(r['distinct_schedules'] == 1 for r in rows),
                   interpretation='1.0 = the policy ignores the surrounding block for this op; '
                                  'lower = its decision depends on block context.')
    (args.out / 'context.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- D1 interaction

def samples(result):
    return (result.get('latency') or {}).get('samples_ns') or []


def validated(result):
    return result.get('status') == 'measured' and (result.get('correctness') or {}).get('status') == 'passed'


def cmd_interaction(args):
    rows = []
    for path in sorted(args.runs.glob('**/artifacts/*/*/summary.json')):
        summary = json.loads(path.read_text())
        res = summary['results']
        base = res.get('mlir_baseline', {})
        full = res.get('manual_full_lenient', {})
        parts = {k[len('manual_only_'):]: v for k, v in res.items() if k.startswith('manual_only_')}
        if not (validated(base) and validated(full) and parts and all(validated(v) for v in parts.values())):
            rows.append(dict(block=summary['workload'], status='incomplete_or_unvalidated'))
            continue
        b, f = samples(base), samples(full)
        p = [samples(v) for v in parts.values()]

        def index(b_med, f_med, p_meds):
            return math.log(b_med / f_med) - sum(math.log(b_med / m) for m in p_meds)
        point = index(stats.median(b), stats.median(f), [stats.median(x) for x in p])
        lo, hi = stats.bootstrap(lambda rs: index(rs[0], rs[1], rs[2:]), [b, f, *p], n=args.bootstrap)
        row = dict(block=summary['workload'], status='ok', groups=len(parts),
                   speedup_full=round(stats.median(b) / stats.median(f), 4),
                   product_of_parts=round(math.exp(sum(math.log(stats.median(b) / stats.median(x)) for x in p)), 4),
                   interaction_index=round(point, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                   verdict='synergy' if lo > 0 else 'interference' if hi < 0 else 'independent')
        nf = res.get('manual_no_fusion', {})
        if validated(nf):
            g = math.log(stats.median(samples(nf)) / stats.median(f))
            glo, ghi = stats.bootstrap(lambda rs: math.log(rs[0] / rs[1]), [samples(nf), f], n=args.bootstrap)
            row.update(fusion_log_gain=round(g, 4), fusion_ci_low=round(glo, 4), fusion_ci_high=round(ghi, 4))
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'interaction.csv', rows)
    ok = [r for r in rows if r.get('status') == 'ok']
    print(f'D1: {len(ok)} blocks analysed; ' + ', '.join(
        f"{v}={sum(r['verdict'] == v for r in ok)}" for v in ('synergy', 'independent', 'interference')))


# --------------------------------------------------------------------------- D3 composition / D4 cache

def read_results(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def cmd_composition(args):
    rows = read_results(args.all_results)
    index = json.loads((args.catalog / 'blocks_index.json').read_text())['blocks']
    time_of = {}
    for r in rows:                     # one entry per config AND repeat (repeats are not mixed)
        if r['status'] == 'measured' and r['correctness'] == 'passed':
            time_of[(f"{r['config']}/r{r['repeat']}", r['block'], r['method'])] = float(r['median_ns'])
    configs = sorted({f"{r['config']}/r{r['repeat']}" for r in rows})
    out = []
    for x, rel in index.items():
        for comp in rel['compositions']:
            y, z = comp['producer_part'], comp['consumer_part']
            for config in configs:
                for method in METHODS:
                    t = [time_of.get((config, n, method)) for n in (x, y, z)]
                    b = [time_of.get((config, n, 'mlir_baseline')) for n in (x, y, z)]
                    if None in t or None in b:
                        continue
                    out.append(dict(config=config, whole=x, producer_part=y, consumer_part=z, method=method,
                                    time_whole_ns=t[0], time_parts_ns=t[1] + t[2],
                                    composition_gain=round((t[1] + t[2]) / t[0], 4),
                                    speedup_measured=round(b[0] / t[0], 4),
                                    speedup_predicted=round((b[1] + b[2]) / (t[1] + t[2]), 4)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'composition.csv', out)
    print(f'D3: {len(out)} (composition, method, config) rows')


def cmd_cache(args):
    rows = read_results(args.all_results)
    by_key = {}
    for r in rows:
        if r['status'] == 'measured' and r['correctness'] == 'passed':
            by_key[(r['threads'], r['repeat'], r['cache'], r['block'], r['method'])] = float(r['median_ns'])
    out = []
    for (threads, repeat, cache, block, method), warm in by_key.items():
        if cache != 'warm':
            continue
        cold = by_key.get((threads, repeat, 'cold', block, method))
        bw = by_key.get((threads, repeat, 'warm', block, 'mlir_baseline'))
        bc = by_key.get((threads, repeat, 'cold', block, 'mlir_baseline'))
        if cold is None or bw is None or bc is None:
            continue
        out.append(dict(threads=threads, repeat=repeat, block=block, method=method,
                        warm_ns=warm, cold_ns=cold, cold_over_warm=round(cold / warm, 4),
                        speedup_warm=round(bw / warm, 4), speedup_cold=round(bc / cold, 4)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'cache.csv', out)
    print(f'D4: {len(out)} (block, method) warm/cold pairs')


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or ['empty'])
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('prepare', help='[MLIR env] policy schedules and variants (no timing)')
    p.add_argument('--selection', type=Path, help='selection.json written by run_blocks.py')
    p.add_argument('--ablation-only', action='store_true', help='Only blocks marked for ablation in the selection')
    p.add_argument('--context-ops', type=int, help='D2: number of shared model ops to sample')
    p.add_argument('--context-blocks-per-op', type=int, default=4)
    p.add_argument('--blocks', nargs='*')
    p.add_argument('--schedules-only', action='store_true')
    p.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    p.add_argument('--data-dir', type=Path, default=cat.DEFAULT_DATA)
    p.add_argument('--checkpoint', default=str(REPO / 'checkpoints/v5_model_1200.pt'))
    p.add_argument('--config', default=str(REPO / 'config/v5/v5_mlir_rl_v1_paper.json'))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rerun', action='store_true')

    c = sub.add_parser('context', help='D2 analysis')
    c.add_argument('--variants', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)

    i = sub.add_parser('interaction', help='D1 analysis')
    i.add_argument('--runs', type=Path, required=True, help='Folder of ablation runs (searched recursively)')
    i.add_argument('--out', type=Path, required=True)
    i.add_argument('--bootstrap', type=int, default=2000)

    for name in ('composition', 'cache'):
        s = sub.add_parser(name, help=f'{"D3" if name == "composition" else "D4"} analysis')
        s.add_argument('--all-results', type=Path, required=True)
        s.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
        s.add_argument('--out', type=Path, required=True)

    args = parser.parse_args()
    {'prepare': cmd_prepare, 'context': cmd_context, 'interaction': cmd_interaction,
     'composition': cmd_composition, 'cache': cmd_cache}[args.command](args)


if __name__ == '__main__':
    main()
