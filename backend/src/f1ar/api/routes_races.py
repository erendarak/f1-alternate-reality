"""Races the backend can simulate, their verified entry lists and drawable tracks."""

from typing import Any

from fastapi import APIRouter

from f1ar.api.catalog import RaceCatalog


def races_router(catalog: RaceCatalog, skipped: dict[str, str]) -> APIRouter:
    router = APIRouter(prefix="/api/races", tags=["races"])

    @router.get("")
    def list_races() -> dict[str, Any]:
        return {
            "races": [r.summary() for r in catalog.all()],
            "unavailable": [{"race_id": k, "reason": v} for k, v in skipped.items()],
        }

    @router.get("/{race_id}")
    def race(race_id: str) -> dict[str, Any]:
        spec = catalog.get(race_id)
        return {**spec.summary(), "drivers": spec.drivers()}

    @router.get("/{race_id}/track")
    def track(race_id: str) -> dict[str, Any]:
        spec = catalog.get(race_id)
        return {**spec.assets.track_payload, "team_colors": spec.assets.team_colors}

    return router
