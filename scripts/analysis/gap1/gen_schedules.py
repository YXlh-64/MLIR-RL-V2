"""Run a trained v5 checkpoint greedily on benchmarks and save its schedules in exec_data.json format (input for B2).

Read-only use of the agent: same rollout as experiments/02_block_diagnosis/ablation.py::Policy.schedule.
Nothing is compiled or timed here; the time field is 0 (unknown).
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('benchmarks', nargs='+', help='benchmark .mlir files, dirs or globs')
ap.add_argument('--checkpoint', required=True, help='e.g. /scratch/mh8826/checkpoints/v5_model_1200.pt')
ap.add_argument('--config', default='config/v5/v5_mlir_rl_v1_paper.json', help='architecture config of the checkpoint')
ap.add_argument('--max-steps', type=int, default=100000)
ap.add_argument('--out', required=True, help='output JSON, e.g. results/gap1/schedules/exec_data.json')
args = ap.parse_args()

# Config is a singleton read at first import: point it at a private copy (no Dask, scratch results_dir).
cfg = json.load(open(args.config))
cfg.update(dask_node_count=0, results_dir=tempfile.mkdtemp(prefix='gap1_policy_'))
cfg_path = Path(cfg['results_dir']) / 'config.json'
cfg_path.write_text(json.dumps(cfg))
os.environ.update(CONFIG_FILE_PATH=str(cfg_path), DASK_NODES='0', DASK_TRAINING_ONLY='1')

import _run  # noqa: E402,F401  (sets sys.path to the repo)
import torch  # noqa: E402
from _mlir_text import files  # noqa: E402
from rl_autoschedular_v5.actions import ActionSpace  # noqa: E402
from rl_autoschedular_v5.env import Env  # noqa: E402
from rl_autoschedular_v5.full_model import ensure_timed_main_abi  # noqa: E402
from rl_autoschedular_v5.model import HiearchyModel  # noqa: E402
from rl_autoschedular_v5.observation import Observation  # noqa: E402
from rl_autoschedular_v5.state import extract_bench_features_from_code  # noqa: E402

model = HiearchyModel().cpu().eval()
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt, strict=True)

out, n_tpf, n_fail = {}, 0, 0
for f in files(args.benchmarks):
    name = os.path.basename(f)[:-5]
    try:
        feats = extract_bench_features_from_code(name, ensure_timed_main_abi(open(f).read()), 1)
        env = Env()
        state = env.reset([feats], 0)
        with torch.inference_mode():
            for _ in range(args.max_steps):
                idx, _, _ = model.sample(Observation.from_state(state), greedy=True)
                state = env.step(state, ActionSpace.action_by_index(idx[0].cpu(), state))
                if state.terminal:
                    nxt = env.get_next_op_state(state)
                    if nxt is None:
                        break
                    state = nxt
            else:
                raise RuntimeError('max steps exceeded')
        # same key format as Execution.get_code_cache_key
        key = '|'.join(''.join(map(str, op_seq)) for op_seq in state.transformation_history)
        out[name] = {key: 0}
        n_tpf += 'TPF(' in key
        print(name, 'TPF' if 'TPF(' in key else '-', flush=True)
    except Exception as e:  # noqa: BLE001
        n_fail += 1
        print(name, 'FAILED', str(e)[:150], flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(out, open(args.out, 'w'), indent=2)
print(f'{len(out)} schedules ({n_tpf} with TPF, {n_fail} failed) -> {args.out}')
