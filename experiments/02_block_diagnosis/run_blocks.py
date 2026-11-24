#!/usr/bin/env python3
"""Select, translate and run block workloads (experiment 02's run_suite.py).

Steps per invocation:
  1. selection   stratified, seeded sample of blocks from the catalog
                 (strata = model x class x size bucket; duplicates removed by signature),
                 plus D3 composition triples (X = Y + Z). Saved once to
                 results/<tag>/selection.json and REUSED by every later run with the same
                 tag, so all thread counts / cache modes / repeats measure the same blocks.
  2. translation each selected block -> Torch graph (block_to_torch.py); failures recorded.
  3. runs        one experiment-01 run per block (4 methods, + manual variants with --ablation),
                 written as results/<tag>/<config>/r<repeat>/<class>/artifacts/<block>/<UTC>/...
                 i.e. experiment 01's layout, so its run_suite.collect_report() writes
                 comparison.csv / REPORT.md / SCHEDULES.md per class, and its
                 plot_speedups.py works on results/<tag>/<config>/r<repeat>.
  4. collect     results/<tag>/all_results.csv: one long table over every config,
                 repeat, block and method, with block metadata. Input of stats.py,
                 ablation.py (D3/D4) and plot_blocks.py.

Modes that run nothing:
  --list           print strata and the selection
  --prepare-only   also translate and write manifests + readable torch_block.py files
  --collect-only   rebuild all_results.csv from finished runs

Examples (from the repo root):
  python experiments/02_block_diagnosis/run_blocks.py --tag main --list
  python experiments/02_block_diagnosis/run_blocks.py --tag main --prepare-only
  python experiments/02_block_diagnosis/run_blocks.py --tag main --threads 1 --repeat 0
  python experiments/02_block_diagnosis/run_blocks.py --tag main --threads 8 --cache cold --repeat 0
  python experiments/02_block_diagnosis/run_blocks.py --tag main --ablation --variants results/main/variants
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import block_catalog as cat        # noqa: E402
import block_to_torch as b2t       # noqa: E402

RESULTS = HERE / 'results'
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor')
DEFAULT_CHECKPOINT = REPO / 'checkpoints/v5_model_1200.pt'
DEFAULT_CONFIG = REPO / 'config/v5/v5_mlir_rl_v1_paper.json'


# --------------------------------------------------------------------------- selection

def eligible(block, args):
    return (block.get('translator_precheck') == 'ok' and block.get('rl_precheck') == 'ok'
            and block.get('abi_bytes') is not None and block['abi_bytes'] <= args.max_input_mib * 2**20
            and block.get('total_flops') is not None and block['total_flops'] <= args.max_work)


def select(args):
    catalog = json.loads((args.catalog / 'blocks_catalog.json').read_text())['blocks']
    index = json.loads((args.catalog / 'blocks_index.json').read_text())['blocks']
    by_name = {b['name']: b for b in catalog}
    ok = [b for b in catalog if eligible(b, args)]

    # De-duplicate identical computations (same signature) before sampling.
    unique = {}
    for b in sorted(ok, key=lambda b: b['name']):
        unique.setdefault(b['signature'], b)
    strata = collections.defaultdict(list)
    for b in unique.values():
        strata[(b['model'], b['class'], b['size_bucket'])].append(b['name'])

    chosen = {}
    for key in sorted(strata):
        rng = random.Random(f'{args.seed}:{":".join(key)}')
        names = sorted(strata[key])
        for name in rng.sample(names, min(args.per_stratum, len(names))):
            chosen[name] = dict(name=name, role='sample', stratum=list(key))

    # D3 composition triples: every part must be eligible (duplicates allowed here).
    triples = [(x, c['producer_part'], c['consumer_part'])
               for x, rel in sorted(index.items()) for c in rel['compositions']
               if all(n in by_name and eligible(by_name[n], args)
                      for n in (x, c['producer_part'], c['consumer_part']))]
    rng = random.Random(f'{args.seed}:compositions')
    for x, y, z in rng.sample(triples, min(args.compositions, len(triples))):
        for name, role in ((x, 'composition_whole'), (y, 'composition_part'), (z, 'composition_part')):
            chosen.setdefault(name, dict(name=name, role=role, stratum=None))

    # D1 ablation subset: blocks with 2..6 ops, a few per class.
    rng = random.Random(f'{args.seed}:ablation')
    by_class = collections.defaultdict(list)
    for name, entry in chosen.items():
        if entry['role'] == 'sample' and 2 <= by_name[name]['n_ops'] <= 6:
            by_class[by_name[name]['class']].append(name)
    for cls, names in sorted(by_class.items()):
        for name in rng.sample(sorted(names), min(args.ablation_per_class, len(names))):
            chosen[name]['ablation'] = True

    for name, entry in chosen.items():
        b = by_name[name]
        entry.update({k: b[k] for k in ('model', 'class', 'size_bucket', 'n_ops', 'total_flops',
                                        'abi_bytes', 'arithmetic_intensity', 'signature', 'source')})
    return dict(seed=args.seed, per_stratum=args.per_stratum, max_input_mib=args.max_input_mib,
                max_work=args.max_work, catalog_blocks=len(catalog), eligible=len(ok),
                unique_eligible=len(unique), strata=len(strata),
                composition_triples_available=len(triples),
                blocks=sorted(chosen.values(), key=lambda e: e['name']))


def load_or_create_selection(args):
    path = RESULTS / args.tag / 'selection.json'
    if path.exists() and not args.reselect:
        return json.loads(path.read_text()), path
    selection = select(args)
    return selection, path


# --------------------------------------------------------------------------- manifests

def source_path(entry, args):
    """Prefer the repo data folder (experiment 01 convention); fall back to the catalog path."""
    local = args.data_dir / f"{entry['name']}.mlir"
    return local if local.exists() else Path(entry['source'])


def manifest(entry, translation, args, variants):
    inputs = [{k: v for k, v in s.items() if k != 'role'} for s in translation['inputs']]
    manual = []
    if args.ablation and variants:
        for name, v in variants.get('variants', {}).items():
            manual.append(dict(name=name, mlir=v['file']))
    return dict(
        schema_version=1, name=entry['name'], mlir=str(source_path(entry, args)),
        dataset='ops_and_blocks', category=entry['class'], seed=args.seed,
        threads=args.threads, cpu_affinity=None,
        warmup=args.warmup, repeats=args.repeats, correctness_cases=3,
        rtol=args.rtol, atol=args.atol, timeout_seconds=args.timeout_seconds,
        profile_repeats=100, profile_min_seconds=0.25, perf=args.perf,
        pipeline_implementation='rl_autoschedular_v5', inputs=inputs,
        torch=b2t.harness_graph(translation),
        rl=dict(checkpoint=str(args.checkpoint), config=str(args.config),
                implementation='rl_autoschedular_v5', max_steps=100000),
        manual=manual,
        cache_flush_mib=args.flush_mib if args.cache == 'cold' else None,
        block={k: entry.get(k) for k in ('model', 'class', 'size_bucket', 'n_ops', 'role',
                                         'total_flops', 'abi_bytes', 'arithmetic_intensity')},
        experiment='02_block_diagnosis', repeat_index=args.repeat,
        rl_apply=args.rl_apply)


def atomic_write(path, text):
    """Write via a temporary file + rename: several jobs on the same tag (e.g. one per
    class) rewrite the same selection/translation files, and none may read a half file."""
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def config_name(args):
    return f't{args.threads}_{args.cache}' + ('_ablation' if args.ablation else '')


# --------------------------------------------------------------------------- collect

def collect(tag):
    root = RESULTS / tag
    selection = {b['name']: b for b in json.loads((root / 'selection.json').read_text())['blocks']}
    rows = []
    latest = {}                          # newest run per (config, repeat, block): reruns replace
    for summary_path in sorted(root.glob('*/r*/*/artifacts/*/*/summary.json')):
        latest[(summary_path.parents[5].name, summary_path.parents[4].name,
                summary_path.parents[1].name)] = summary_path
    for summary_path in sorted(latest.values()):
        config_dir, repeat_dir = summary_path.parents[5].name, summary_path.parents[4].name
        if config_dir.endswith('_ablation'):
            continue                    # D1 runs are analysed by ablation.py interaction
        threads, cache = config_dir.split('_')[:2]
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue                    # being written by a parallel job; picked up next collect
        block = summary['workload']
        meta = selection.get(block, {})
        for method, result in summary['results'].items():
            latency = result.get('latency') or {}
            compile_ns = sum(v for v in (result.get('compilation') or {}).values() if isinstance(v, (int, float)))
            rows.append(dict(
                config=f'{config_dir}', threads=threads[1:], cache=cache, repeat=repeat_dir[1:],
                block=block, model=meta.get('model'), **{'class': meta.get('class')},
                size_bucket=meta.get('size_bucket'), n_ops=meta.get('n_ops'), role=meta.get('role'),
                method=method, status=result.get('status'),
                correctness=(result.get('correctness') or {}).get('status'),
                median_ns=latency.get('median_ns'), mean_ns=latency.get('mean_ns'),
                stdev_ns=latency.get('stdev_ns'),
                validated_speedup=result.get('validated_speedup_vs_mlir_baseline'),
                observed_ratio=result.get('observed_latency_ratio_vs_mlir_baseline'),
                compile_ns=compile_ns or None,
                total_flops=meta.get('total_flops'), abi_bytes=meta.get('abi_bytes'),
                arithmetic_intensity=meta.get('arithmetic_intensity'),
                reason=result.get('reason', ''),
                **_rl_application(summary_path.parent, method),
                result_json=str(summary_path.parent / 'methods' / method / 'result.json')))
    if not rows:
        print(f'No finished runs under {root}')
        return None
    path = root / 'all_results.csv'
    tmp = path.with_name(f'.all_results.{os.getpid()}.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print(f'Wrote {path} ({len(rows)} rows)')
    return path


def _rl_application(run_dir, method):
    if method != 'mlir_rl':
        return dict(rl_apply='', rl_failed_actions='', rl_skipped_actions='', rl_applied_actions='')
    try:
        mode = json.loads((run_dir / 'manifest.json').read_text()).get('rl_apply', 'strict')
    except (OSError, json.JSONDecodeError):
        mode = 'strict'
    try:
        log = json.loads((run_dir / 'methods' / 'mlir_rl' / 'rl_application.json').read_text())
    except (OSError, json.JSONDecodeError):
        log = {}
    return dict(rl_apply=mode, rl_failed_actions=len(log.get('failed', [])) if log else '',
                rl_skipped_actions=len(log.get('skipped', [])) if log else '',
                rl_applied_actions=log.get('applied', '') if log else '')


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='main', help='Name of this study; results/<tag>/')
    p.add_argument('--catalog', type=Path, default=HERE / 'results' / 'catalog')
    p.add_argument('--data-dir', type=Path, default=cat.DEFAULT_DATA)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--per-stratum', type=int, default=1, help='Blocks per (model, class, size) stratum')
    p.add_argument('--compositions', type=int, default=15, help='D3 triples to include')
    p.add_argument('--ablation-per-class', type=int, default=5, help='D1 blocks per class')
    p.add_argument('--max-input-mib', type=float, default=32, help='Same limit as experiment 01')
    p.add_argument('--max-work', type=float, default=1e8, help='Max estimated FLOPs (experiment 01: 1e8)')
    p.add_argument('--reselect', action='store_true', help='Overwrite an existing selection.json')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--cache', choices=('warm', 'cold'), default='warm')
    p.add_argument('--flush-mib', type=int, default=1024, help='Cold mode: buffer rewritten before each call')
    p.add_argument('--repeat', type=int, default=0, help='Independent repeat index (separate folder)')
    p.add_argument('--methods', nargs='+', choices=METHODS)
    p.add_argument('--ablation', action='store_true', help='Run D1 variants for blocks marked ablation')
    p.add_argument('--variants', type=Path, help='Folder written by ablation.py prepare')
    p.add_argument('--only', nargs='+', help='Restrict to these selected block names')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--timeout-seconds', type=int, default=1200)
    p.add_argument('--perf', action='store_true', help='Collect perf counters (experiment 01 default: off)')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--list', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--rerun', action='store_true')
    p.add_argument('--rl-apply', choices=('strict', 'lenient'), default='strict',
                   help='strict: experiment 01 behaviour (one failed RL action = mlir_rl error). '
                        'lenient: like the RL Env, a failed action is skipped with the rest of its '
                        "op's actions; recorded in methods/mlir_rl/rl_application.json")
    args = p.parse_args()

    if args.collect_only:
        collect(args.tag)
        return 0

    selection, sel_path = load_or_create_selection(args)
    blocks = selection['blocks']
    if args.only:
        blocks = [b for b in blocks if b['name'] in set(args.only)]
    if args.ablation:
        blocks = [b for b in blocks if b.get('ablation')]
        if not args.variants:
            raise SystemExit('--ablation needs --variants (run ablation.py prepare first)')

    roles = collections.Counter(b['role'] for b in blocks)
    print(f"Catalog {selection['catalog_blocks']} blocks; eligible {selection['eligible']}; "
          f"unique {selection['unique_eligible']}; strata {selection['strata']}")
    print(f"Selected {len(blocks)} blocks: {dict(roles)}; ablation-marked: "
          f"{sum(1 for b in selection['blocks'] if b.get('ablation'))}")
    if args.list:
        by_class = collections.Counter((b['class'], b['size_bucket']) for b in blocks)
        for key, n in sorted(by_class.items()):
            print(f'  {key[0]:12s} {key[1]:4s} {n}')
        for b in blocks:
            print(f"  {b['name']:40s} {b['class']:12s} ops={b['n_ops']:<3} {b['role']}"
                  + ('  [ablation]' if b.get('ablation') else ''))
        return 0

    # Translate (stdlib only) and save the selection with translation status.
    root = RESULTS / args.tag
    trans_dir = root / 'translations'
    trans_dir.mkdir(parents=True, exist_ok=True)
    translations = {}
    for entry in selection['blocks']:
        t = b2t.translate_file(source_path(entry, args))
        entry['translation'] = t['status']
        entry['translation_reason'] = t.get('reason')
        atomic_write(trans_dir / f"{entry['name']}.torch.json", json.dumps(t, indent=1))
        if t['status'] == 'ok':
            atomic_write(trans_dir / f"{entry['name']}.py", b2t.render_python(entry['name'], t))
            translations[entry['name']] = t
    atomic_write(sel_path, json.dumps(selection, indent=1))
    ok = sum(1 for e in selection['blocks'] if e['translation'] == 'ok')
    print(f'Translated {ok}/{len(selection["blocks"])} (details: {trans_dir})')

    config_dir = root / config_name(args) / f'r{args.repeat}'
    if args.prepare_only:
        mdir = config_dir / 'manifests'
        mdir.mkdir(parents=True, exist_ok=True)
        for entry in blocks:
            if entry['name'] in translations:
                variants = _variants(args, entry['name'])
                (mdir / f"{entry['name']}.json").write_text(json.dumps(
                    manifest(entry, translations[entry['name']], args, variants), indent=1))
        print(f'Manifests: {mdir} (run one with: python {HERE.name}/block_experiment.py run <manifest>)')
        return 0

    import block_experiment                      # imports experiment 01 (needs numpy)
    import run_suite                             # experiment 01's report writer
    failed = 0
    for k, entry in enumerate(blocks, 1):
        name = entry['name']
        if name not in translations:
            print(f'[{k}/{len(blocks)}] {name}: skipped, translation {entry["translation"]}: '
                  f'{entry.get("translation_reason")}')
            continue
        variants = _variants(args, name)
        if args.ablation and not variants.get('variants'):
            print(f'[{k}/{len(blocks)}] {name}: skipped, no prepared variants')
            continue
        dest = config_dir / entry['class'] / 'artifacts' / name
        if not args.rerun and _done(dest, args.rl_apply):
            print(f'[{k}/{len(blocks)}] {name}: already done')
            continue
        methods = list(args.methods or (('mlir_baseline', 'mlir_rl', 'torch_eager') if args.ablation else METHODS))
        print(f'[{k}/{len(blocks)}] {name} ({entry["class"]}, {entry["n_ops"]} ops) -> {dest}', flush=True)
        failed |= block_experiment.run(manifest(entry, translations[name], args, variants), methods, dest)
        try:                                     # parallel shards share class folders
            run_suite.collect_report(config_dir / entry['class'])
        except Exception as error:               # a summary.json mid-write elsewhere; report job redoes it
            print(f'  (per-class report skipped this time: {type(error).__name__}: {error})')
    if not args.ablation:
        collect(args.tag)
    return 1 if failed else 0


def _done(dest, rl_apply):
    """A block is done if its latest run finished with the same RL application mode
    (earlier runs in another mode are kept in their own timestamped folders)."""
    runs = sorted(p.parent for p in dest.glob('*/summary.json'))
    if not runs:
        return False
    try:
        mode = json.loads((runs[-1] / 'manifest.json').read_text()).get('rl_apply', 'strict')
    except (OSError, json.JSONDecodeError):
        mode = 'strict'
    return mode == rl_apply


def _variants(args, name):
    if not args.variants:
        return {}
    path = args.variants / name / 'variants.json'
    return json.loads(path.read_text()) if path.exists() else {}


if __name__ == '__main__':
    raise SystemExit(main())
