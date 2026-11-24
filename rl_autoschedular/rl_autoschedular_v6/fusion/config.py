"""Fusion-layer switches, read from the `fusion` section of the v6 config (design doc §6).

Every module is off by default, so a config without a `fusion` section (or
with `{}`) reproduces v5 behaviour exactly.

Execution-cache isolation: the per-run cache is `<results_dir>/logs/exec_data.json`
(FileLogger), and its key is bench name + action string. A prefused program
must never reuse times measured on the unfused one, so every fusion config gets
its own `results_dir`, and a shared `main_exec_data_file` is rejected while
prefusion is on.
"""

from dataclasses import dataclass, field, fields
from functools import cache
from typing import Any


@dataclass(frozen=True)
class FusionConfig:
    prefuse: bool = False          # M2: green pre-fusion of every loaded benchmark
    producer_select: bool = False  # M4: TF gets the best data producer instead of producers[-1]
    tile_guard: bool = False       # M5: mask TF tile sizes with high recomputation / L2 overflow
    tile_guard_params: dict = field(default_factory=dict)  # M5 thresholds (validated in tile_guard.py)
    units: dict = field(default_factory=dict)  # M3 offline builder parameters (validated in units.py)
    sampling_weights: str = ''     # JSON {benchmark: weight}; '' = uniform, as v5 (fusion/sampling.py)

    # Switches that make Benchmarks.__init__ call apply_fusion. sampling_weights is not one:
    # it is read where the iteration's benchmarks are drawn (fusion/sampling.py).
    MODULE_FLAGS = ('prefuse', 'producer_select', 'tile_guard')

    @property
    def any_enabled(self) -> bool:
        return any(getattr(self, flag) for flag in self.MODULE_FLAGS)

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> 'FusionConfig':
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f'unknown fusion config keys: {sorted(unknown)} (known: {sorted(known)})')
        return cls(**section)


def check_cache_isolation(fusion: FusionConfig, main_exec_data_file: str) -> None:
    """Reject a shared execution cache while the programs differ from the unfused ones."""
    if fusion.prefuse and main_exec_data_file:
        raise ValueError(
            'fusion.prefuse is on but main_exec_data_file is set: that cache holds times of '
            'unfused programs under the same keys. Use an empty main_exec_data_file.'
        )


@cache
def load_fusion_config() -> FusionConfig:
    """FusionConfig from `Config().fusion`, with the execution-cache guard.
    Cached: Config is a process-wide singleton, and Env asks for it per op."""
    from rl_autoschedular_v6.utils.config import Config

    cfg = Config()
    fusion = FusionConfig.from_dict(cfg.fusion)
    check_cache_isolation(fusion, cfg.main_exec_data_file)
    return fusion
