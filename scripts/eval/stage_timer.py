#!/usr/bin/env python3
"""Stage-timer for full-model execution: reports where wall-clock goes.

Stages: file read -> ABI adapt -> Module.parse -> bufferize
-> lower (pass pipeline) -> ExecutionEngine(opt_level=3) -> invoke x2.
Flushes after every stage so a hang is attributable. Usage:
    CONFIG_FILE_PATH=... AUTOSCHEDULER_IMPL=rl_autoschedular_v5 \
    python scripts/eval/stage_timer.py --model-file <path> --bench <name> [--timeout S]
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--timeout", type=int, default=0)
    args = parser.parse_args()

    if args.timeout:
        import signal

        def _handler(signum, frame):
            raise TimeoutError(f"stage timer expired after {args.timeout}s")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(args.timeout)

    from utils.implementation import get_autoschedular_impl, import_autoschedular_module

    IMPL = get_autoschedular_impl()
    exec_mod = import_autoschedular_module("execution", IMPL)
    Execution = exec_mod.Execution
    _EXEC_PASS_PIPELINE = exec_mod._EXEC_PASS_PIPELINE
    full_model = import_autoschedular_module("full_model", IMPL)
    transforms = import_autoschedular_module("transforms", IMPL)
    from mlir.ir import Context, Module  # noqa: E402
    from mlir.passmanager import PassManager  # noqa: E402
    from mlir.execution_engine import ExecutionEngine  # noqa: E402

    t0 = time.time()
    code = open(args.model_file).read()
    log(f"read {len(code) / 1e6:.1f}MB ({time.time() - t0:.1f}s)")

    t0 = time.time()
    code = full_model.materialize_elided_resources(code)
    code = full_model.ensure_timed_main_abi(code)
    log(f"ABI adapt ({time.time() - t0:.1f}s)")

    t0 = time.time()
    with Context() as ctx:
        module = Module.parse(code)
        log(f"Module.parse ({time.time() - t0:.1f}s)")

        t0 = time.time()
        transforms.transform_bufferize_and_lower_v(module)
        log(f"bufferize ({time.time() - t0:.1f}s)")

        t0 = time.time()
        pm = PassManager.parse(_EXEC_PASS_PIPELINE, ctx)
        inputs, outs = Execution._create_params(module)
        test_args = Execution._convert_to_args(inputs, outs)
        pm.run(module.operation)
        log(f"lower ({time.time() - t0:.1f}s)")

        t0 = time.time()
        test_args = Execution._convert_to_args(inputs, outs)
        engine = ExecutionEngine(
            module, opt_level=3,
            shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
        )
        log(f"JIT opt_level=3 ({time.time() - t0:.1f}s)")

        t0 = time.time()
        engine.invoke("main", *test_args)
        d1 = outs.delta
        outs.free_outputs()
        engine.invoke("main", *test_args)
        d2 = outs.delta
        outs.free_outputs()
        log(f"invoke x2: {d1 / 1e6:.1f}ms, {d2 / 1e6:.1f}ms ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
