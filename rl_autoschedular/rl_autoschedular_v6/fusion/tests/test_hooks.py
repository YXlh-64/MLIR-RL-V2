"""Integration: FusionConfig parsing, the cache guard, and apply_fusion (features re-extracted)."""

import pytest

from conftest import FIXTURE_NAMES, ast_dump, fixture_features, needs_ast_dumper
from rl_autoschedular_v6.fusion.config import FusionConfig, check_cache_isolation
from rl_autoschedular_v6.fusion.hooks import apply_fusion
from rl_autoschedular_v6.fusion.prefuse import prefuse_code


def test_config_defaults_off_and_rejects_unknown_keys():
    assert not FusionConfig.from_dict({}).any_enabled
    assert FusionConfig.from_dict({'prefuse': True}).any_enabled
    with pytest.raises(ValueError):
        FusionConfig.from_dict({'prefusion': True})


def test_cache_guard():
    check_cache_isolation(FusionConfig(prefuse=True), '')
    check_cache_isolation(FusionConfig(), 'shared_cache.json')
    with pytest.raises(ValueError):
        check_cache_isolation(FusionConfig(prefuse=True), 'shared_cache.json')


def test_all_off_returns_same_benchmarks():
    benches = [fixture_features(n) for n in FIXTURE_NAMES]
    assert apply_fusion(benches, FusionConfig()) == benches


@needs_ast_dumper
@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_prefuse_reextracts_features(name, tmp_path):
    """MUST FIX 1: after prefusion, tags and op count match the fused file, not the original."""
    original = fixture_features(name, root_exec_time=1234)
    [fused] = apply_fusion([original], FusionConfig(prefuse=True))

    fused_code, report = prefuse_code(original.code)
    path = tmp_path / f'{name}.mlir'
    path.write_text(fused_code)
    names, _ = ast_dump(str(path))

    assert fused.operation_tags == list(names)
    assert len(fused.operations) == report.ops_after
    assert fused.root_exec_time == 1234                     # R12: baseline of the original program
    assert len(original.operation_tags) == report.ops_before  # input left untouched
