"""Selector registry. All arms share one signature so train.py stays arm-agnostic."""
from .coverage import coverage_select
from .baselines import uniform_select, length_select, placebo_select, d4_select

SELECTORS = {
    "uniform":  uniform_select,
    "length":   length_select,
    "ontology": coverage_select,
    "placebo":  placebo_select,
    "d4":       d4_select,
}

def get_selector(name: str):
    if name not in SELECTORS:
        raise KeyError(f"unknown arm '{name}'. options: {sorted(SELECTORS)}")
    return SELECTORS[name]
