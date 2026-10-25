#!/usr/bin/env python3
"""Baseline timing for full-model .mlir files (V5 execution API).

The legacy scripts/baseline/get_base.py calls
``execute_code(code_string, ...)`` — the pre-V5 API. V5's
``Execution.execute_code`` takes a parsed ``Module``. This driver uses the V5
API directly: read file -> ABI-adapt (no-op for conforming files) ->
Module.parse -> isolated execution.

One process handles all files sequentially; each *execution* runs in an
isolated child process. Files that fail parse or time out record -1.
Completed entries are skipped on re-run (resume via the output JSON).

Usage:
    export CONFIG_FILE_PATH=config/v5/v5_mlir_rl_v1_paper_fullmodel.json
    export AUTOSCHEDULER_IMPL=rl_autoschedular_v5
    python scripts/baseline/baseline_fullmodels.py \\
        --models-dir data/full_models --output /tmp/baselines.json --timeout 600
"""
import argparse
import glob
import json
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')


def _timeout_handler(signum, frame):
    raise TimeoutError()


def main():
    parser = argparse.ArgumentParser(description="Baseline timing for full-model files.")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*_linalg.mlir")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    from utils.implementation import get_autoschedular_impl, import_autoschedular_module

    IMPL = get_autoschedular_impl()
    Execution = import_autoschedular_module("execution", IMPL).Execution
    full_model = import_autoschedular_module("full_model", IMPL)
    from mlir.ir import Context, Module  # noqa: E402

    exe_cache = os.path.join(os.path.dirname(args.output), ".baseline_exec_cache.json")
    if not os.path.exists(exe_cache):
        json.dump({}, open(exe_cache, "w"))
    os.environ["MAX_EXEC_TIMEOUT"] = str(args.timeout)
    os.environ["COMPILE_MIN_TIMEOUT"] = str(args.timeout)

    Execution(exe_cache, None)
    exe = Execution()

    output = {}
    if os.path.exists(args.output):
        output = json.load(open(args.output))

    files = sorted(glob.glob(os.path.join(args.models_dir, args.pattern)))
    print(f"Files: {len(files)}, timeout={args.timeout}s", flush=True)
    for f in files:
        bench = os.path.basename(f)[:-len(".mlir")]
        if output.get(bench, -1) and output.get(bench, -1) > 0:
            print(f"  {bench}: cached {output[bench]}", flush=True)
            continue
        try:
            with open(f) as fh:
                code = fh.read()
            code = full_model.materialize_elided_resources(code)
            code = full_model.ensure_timed_main_abi(code)
            with Context():
                module = Module.parse(code)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(args.timeout)
            try:
                et, ok, _, err = exe.execute_code(module, bench, [], None)
                et = et if ok else -1
                if not ok:
                    print(f"  {bench}: FAILED", flush=True)
                    print(f"  --- error ---\n{err}\n  --- end error ---", flush=True)
            finally:
                signal.alarm(0)
        except TimeoutError:
            print(f"  {bench}: TIMEOUT after {args.timeout}s", flush=True)
            et = -1
        except Exception as e:
            print(f"  {bench}: ERROR {str(e)[:150]}", flush=True)
            et = -1
        output[bench] = et
        json.dump(output, open(args.output, "w"), indent=2)
        print(f"  {bench}: {et}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
