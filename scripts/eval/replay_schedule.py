#!/usr/bin/env python3
"""Replay a logged greedy schedule, optionally skipping action classes.

Parses the per-op action lists an eval run prints
(`  - <op_tag>: ['I(..)', 'T(..)', ...]`), rebuilds Action objects via
ActionSpace.action_from_str with live Env states (same feature freshness as
rollout), applies them to the full Module, saves it, and compile-checks.

Uses:
  - bisect a compile-breaking schedule (with --skip-symbols to test suspects)
  - test whether a schedule class (e.g. fusion) is the culprit

Usage:
    CONFIG_FILE_PATH=... AUTOSCHEDULER_IMPL=... \
    python scripts/eval/replay_schedule.py --log logs/eval_fullmodel_X.out \\
        --checkpoint <ckpt> --out /tmp/replay.mlir [--skip-symbols TPF] [--full]
    --full also runs lower + JIT + one invoke (slow on huge modules).
"""
import argparse
import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')
os.environ['DASK_TRAINING_ONLY'] = '1'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_schedule(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', open(path, errors='replace').read())
    sched = {}
    for m in re.finditer(r"^\s*-\s*(\S+): \[(.*?)\]\s*$", text, re.M):
        tag = m.group(1)
        strs = [s.strip().strip("'\"") for s in m.group(2).split(',') if s.strip()]
        # regroup split params: action strings contain commas inside parens
        actions, buf, depth = [], "", 0
        for s in m.group(2).split(','):
            buf += (',' if buf else '') + s.strip()
            depth += buf.count('(') - buf.count(')')
            if depth <= 0 and buf:
                actions.append(buf.strip().strip("'\""))
                buf, depth = "", 0
        if buf:
            actions.append(buf.strip().strip("'\""))
        sched[tag] = actions
    return sched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-symbols", default="", help="comma-separated symbols to drop, e.g. TPF")
    parser.add_argument("--full", action="store_true", help="also lower + JIT + invoke")
    args = parser.parse_args()

    import torch
    from utils.implementation import get_autoschedular_impl, import_autoschedular_module

    IMPL = get_autoschedular_impl()
    Model = import_autoschedular_module("model", IMPL).HiearchyModel
    device = import_autoschedular_module("", IMPL).device
    ActionSpace = import_autoschedular_module("actions", IMPL).ActionSpace
    Benchmarks = import_autoschedular_module("benchmarks", IMPL).Benchmarks
    Env = import_autoschedular_module("env", IMPL).Env
    exec_mod = import_autoschedular_module("execution", IMPL)
    from mlir.ir import Context, Module  # noqa: E402

    skip = {s.strip() for s in args.skip_symbols.split(',') if s.strip()}
    sched = parse_schedule(args.log)
    log(f"parsed {len(sched)} ops from log (skipping symbols: {skip or 'none'})")

    torch.set_grad_enabled(False)
    benchs = Benchmarks(is_training=False)
    assert len(benchs) == 1, f"expected 1 bench, got {len(benchs)}"
    model = Model().to(device)  # device placement only; no sampling in replay
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)

    env = Env()
    state = env.reset(benchs, 0)
    used, skipped, missing = 0, 0, 0
    while True:
        tag = state.operation_tag
        strs = sched.get(tag)
        if strs is None:
            missing += 1
        else:
            for s in strs:
                if s.split('(')[0] in skip:
                    skipped += 1
                    continue
                try:
                    action = ActionSpace.action_from_str(state, s)
                except Exception as e:
                    log(f"  from_str failed for {tag} {s}: {str(e)[:100]}")
                    continue
                state = env.step(state, action)
                used += 1
                if state.terminal:
                    break
        if state.terminal:
            nxt = env.get_next_op_state(state)
            if nxt is None:
                break
            state = nxt
        else:
            # op had no logged actions or all skipped: advance without acting
            state.terminal = True
            nxt = env.get_next_op_state(state)
            if nxt is None:
                break
            state = nxt
    log(f"replay done: {used} applied, {skipped} skipped, {missing} ops w/o logged schedule")

    t0 = time.time()
    module, _ = env._Env__apply_sequence(state.transformation_history)
    code = str(module)
    open(args.out, "w").write(code)
    log(f"transformed module saved ({len(code) / 1e6:.1f}MB, {time.time() - t0:.0f}s)")

    t0 = time.time()
    with Context() as ctx:
        m2 = Module.parse(code)
        log(f"re-parse ({time.time() - t0:.1f}s)")
        t0 = time.time()
        from rl_autoschedular_v5.transforms import transform_bufferize_and_lower_v
        transform_bufferize_and_lower_v(m2)
        log(f"bufferize ({time.time() - t0:.1f}s)")
        if not args.full:
            log("compile-check OK (bufferize passed; --full for lower+JIT)")
            return 0
        from mlir.passmanager import PassManager  # noqa: E402
        from mlir.execution_engine import ExecutionEngine  # noqa: E402
        t0 = time.time()
        pm = PassManager.parse(exec_mod._EXEC_PASS_PIPELINE, ctx)
        Execution = exec_mod.Execution
        inputs, outs = Execution._create_params(m2)
        test_args = Execution._convert_to_args(inputs, outs)
        pm.run(m2.operation)
        log(f"lower ({time.time() - t0:.1f}s)")
        t0 = time.time()
        engine = ExecutionEngine(m2, opt_level=3,
                                 shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","))
        log(f"JIT ({time.time() - t0:.1f}s)")
        t0 = time.time()
        engine.invoke("main", *test_args)
        d = outs.delta
        outs.free_outputs()
        log(f"invoke: {d / 1e6:.1f}ms ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
