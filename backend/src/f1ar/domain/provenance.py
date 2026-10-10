"""Data provenance categories (plan §3.4).

Every value in the world belongs to exactly one of these. They are never mixed silently:
real data, model estimates, design assumptions and simulation output stay distinguishable
in storage, reports and the UI.
"""

from enum import StrEnum


class SourceType(StrEnum):
    HISTORICAL_VERIFIED = "historical_verified"
    """Checkable records from a named source: names, numbers, teams, grid, results, laps."""

    ESTIMATED_FROM_DATA = "estimated_from_data"
    """Derived from real data by a model or fit (pace baselines, degradation curves)."""

    EDITORIAL_ASSUMPTION = "editorial_assumption"
    """Design choices with no measurement behind them (persona values, templates)."""

    SIMULATED_DYNAMIC = "simulated_dynamic"
    """Produced by the simulation as it runs (fatigue, morale, alternate results)."""

    LEARNED_PARAMETER = "learned_parameter"
    """Produced by ML training (policy weights, learned embeddings)."""
