#!/usr/bin/env python3
"""Reproduce gpt2's transformed module and stage-time its compile.

Replays the exact greedy rollout (same ckpt, greedy=True -> deterministic),
rebuilds the transformed Module via Env._Env__apply_sequence, saves it, then
times bufferize / lower / JIT / invoke with flushed prints so the failing
stage is attributable. See scripts/eval/stage_timer.py for the raw path.
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')
os.environ['DASK_TRAINING_ONLY'] = '1'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="where to save transformed module")
    args = parser.parse_args()

    import torch
    from utils.implementation import get_autoschedular_impl, import_autoschedular_module

    IMPL = get_autoschedular_impl()
    Model = import_autoschedular_module("model", IMPL).HiearchyModel
    device = import_autoschedular_module("", IMPL).device
    Observation = import_autoschedular_module("observation", IMPL).Observation
    ActionSpace = import_autoschedular_module("actions", IMPL).ActionSpace
    Benchmarks = import_autoschedular_module("benchmarks", IMPL).Benchmarks
    Env = import_autoschedular_module("env", IMPL).Env
    exec_mod = import_autoschedular_module("execution", IMPL)
    Execution = exec_mod.Execution
    _EXEC_PASS_PIPELINE = exec_mod._EXEC_PASS_PIPELINE
    full_model = import_autoschedular_module("full_model", IMPL)
    transforms = import_autoschedular_module("transforms", IMPL)
    from mlir.ir import Context, Module  # noqa: E402
    from mlir.passmanager import PassManager  # noqa: E402
    from mlir.execution_engine import ExecutionEngine  # noqa: E402

    torch.set_grad_enabled(False)
    benchs = Benchmarks(is_training=False)
    assert len(benchs) == 1, f"expected 1 bench, got {len(benchs)}"
    model = Model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    log("model loaded, starting greedy rollout")

    t0 = time.time()
    env = Env()
    state = env.reset(benchs, 0)
    obs = Observation.from_state(state)
    steps = 0
    while not state.terminal:
        actions_index, _, _ = model.sample(obs.to(device), greedy=True)
        action = ActionSpace.action_by_index(actions_index.cpu().reshape(-1), state)
        state = env.step(state, action)
        obs = Observation.from_state(state)
        steps += 1
        if state.terminal:
            nxt = env.get_next_op_state(state)
            if nxt is None:
                break
            state = nxt
            obs = Observation.from_state(state)
        if steps % 1000 == 0:
            log(f"rollout step {steps}")
    log(f"rollout done: {steps} steps in {time.time() - t0:.0f}s")

    t0 = time.time()
    module, _ = env._Env__apply_sequence(state.transformation_history)
    code = str(module)
    open(args.out, "w").write(code)
    log(f"transformed module saved ({len(code) / 1e6:.1f}MB, {time.time() - t0:.0f}s) -> {args.out}")

    t0 = time.time()
    with Context() as ctx:
        m2 = Module.parse(code)
        log(f"re-parse transformed ({time.time() - t0:.1f}s)")
        t0 = time.time()
        from rl_autoschedular_v5.transforms import transform_bufferize_and_lower_v
        transform_bufferize_and_lower_v(m2)
        log(f"bufferize ({time.time() - t0:.1f}s)")
        t0 = time.time()
        pm = PassManager.parse(exec_mod._EXEC_PASS_PIPELINE, m2.context)
        Execution = exec_mod.Execution
        inputs, outs = Execution._create_params(m2)
        args_ = Execution._convert_to_args(inputs, outs)
        pm.run(m2.operation)
        log(f"lower ({time.time() - t0:.1f}s)")
        t0 = time.time()
        engine = ExecutionEngine(m2, opt_level=3,
                                 shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","))
        log(f"JIT ({time.time() - t0:.1f}s)")
        t0 = time.time()
        engine.invoke("main", *args_)
        d = outs.delta
        outs.free_outputs()
        log(f"invoke: {d / 1e6:.1f}ms ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
