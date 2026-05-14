"""MCP tool: km_refresh_action_templates.

In-process wrapper around scripts/capture_km_action_templates.py. Drives the
KM editor's Insert Action menu to (re)capture canonical action XML for every
action under the requested categories, and merges results into
src/server/data/km_action_templates.json.

The scrape takes over the KM editor for several seconds per action — running
it requires explicit `confirm=true` so a stray call never hijacks an active
editing session. Returns a structured diff (added / removed / unchanged) so
agents can verify what changed without re-reading the data file.

@stable
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ._action_templates import load_templates

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "km_action_templates.json"


async def km_refresh_action_templates(
    macro_id: Annotated[
        str,
        Field(
            description=(
                "Scratch macro UUID the scrape will append and delete actions in. "
                "The macro is cleared at start and end. It must exist; create it "
                "first with km_create_macro if needed."
            ),
            min_length=1,
            max_length=64,
        ),
    ],
    confirm: Annotated[
        bool,
        Field(
            description=(
                "Required: explicitly set to true to run the scrape. The KM editor "
                "is taken over for the duration; do not run this while a user is "
                "actively editing macros."
            ),
        ),
    ] = False,
    categories: Annotated[
        list[str] | None,
        Field(
            description=(
                "Subset of Insert Action submenu names to walk. None = full catalog. "
                "Use this to refresh just one category after a KM update."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(
            description="Stop after N captures (0 = no limit). Useful for smoke tests.",
            ge=0,
            le=500,
        ),
    ] = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    """Refresh km_action_templates.json by driving KM's editor menu.

    Failure modes:
    - VALIDATION_ERROR: confirm=false (default) or empty macro_id
    - SCRAPE_FAILED: AppleScript or osascript subprocess returned non-zero
    - SCRATCH_MACRO_NOT_FOUND: macro_id does not resolve in KM
    """
    correlation_id = str(uuid.uuid4())
    if ctx:
        await ctx.info(
            f"km_refresh_action_templates confirm={confirm} macro={macro_id!r} "
            f"limit={limit} [ID: {correlation_id}]",
        )

    if not confirm:
        return {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "confirm=true is required to run the scrape.",
                "recovery_suggestion": (
                    "This tool takes over the KM editor. Pass confirm=true once you "
                    "have a scratch macro and you're not actively editing."
                ),
            },
        }

    before = set(load_templates().keys())
    try:
        diff = await asyncio.to_thread(
            _run_scrape, macro_id, tuple(categories) if categories else None, limit,
        )
    except Exception as exc:
        logger.exception("km_refresh_action_templates failed [ID: %s]", correlation_id)
        return {
            "success": False,
            "error": {
                "code": "SCRAPE_FAILED",
                "message": str(exc),
                "recovery_suggestion": (
                    "Check Accessibility permission for osascript and confirm the "
                    "macro_id exists. Re-run with limit=3 to localize the failure."
                ),
            },
        }
    load_templates.cache_clear()
    after = set(load_templates().keys())
    return {
        "success": True,
        "data": {
            "diff": diff,
            "added": sorted(after - before),
            "unchanged": sorted(after & before),
            "removed": sorted(before - after),
            "total_after": len(after),
        },
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "correlation_id": correlation_id,
        },
    }


def _run_scrape(
    macro_id: str,
    categories: tuple[str, ...] | None,
    limit: int,
) -> dict[str, Any]:
    # Imported lazily so the heavy import (subprocess wiring) doesn't run at
    # module load — keeps the tool registry boot fast.
    from scripts.capture_km_action_templates import (  # noqa: PLC0415
        CATEGORIES_DEFAULT,
        run,
    )

    before_raw = json.loads(_DEFAULT_OUT.read_text()) if _DEFAULT_OUT.exists() else {}
    run(
        out_path=_DEFAULT_OUT,
        macro_id=macro_id,
        categories=categories or CATEGORIES_DEFAULT,
        limit=limit,
    )
    after_raw = json.loads(_DEFAULT_OUT.read_text())
    return {
        "captured_before": len(before_raw),
        "captured_after": len(after_raw),
    }
