"""The evaluation set — single source of truth (`config/fusion/eval_set.json`).

Read by:
- the units leakage guard (`units.py`): every listed model is excluded from training data;
- `scripts/fusion_data/eval_set_tools.py`: the eval benchmark JSON (runnable models only)
  and the model-survey comparison set;
- the config tests, which check that the eval configs evaluate exactly the runnable models.
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass(frozen=True)
class EvalModel:
    name: str                  # e.g. "resnet50"
    file: str                  # stem in data/full_models, e.g. "resnet50_linalg"
    runnable: bool             # evaluated in v6 today
    reason: str                # why not runnable (empty if runnable)
    survey: Optional[dict]     # model_survey build recipe {category, kind, repo}, None if not buildable


def load_eval_set(path: str) -> list[EvalModel]:
    with open(path) as f:
        return [EvalModel(**entry) for entry in json.load(f)['models']]


def excluded_names(path: str) -> set[str]:
    """Model names that must never become training units."""
    return {m.name for m in load_eval_set(path)}


def runnable_models(path: str) -> list[EvalModel]:
    return [m for m in load_eval_set(path) if m.runnable]
