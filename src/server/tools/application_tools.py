"""Application control tool — launch, quit, activate, query macOS apps.

Thin MCP adapter over `src.applications.app_controller.AppController`.
All business logic (security checks, AppleScript escaping, state caching)
lives in that class; this module only translates between MCP's JSON-dict
contract and the controller's `Either[KMError, AppOperationResult]` API.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...applications.app_controller import (
    AppController,
    AppIdentifier,
    LaunchConfiguration,
)
from ...core.types import Duration

logger = logging.getLogger(__name__)

_app_controller: AppController | None = None


def _get_controller() -> AppController:
    global _app_controller
    if _app_controller is None:
        _app_controller = AppController()
    return _app_controller


def _build_app_id(identifier: str) -> AppIdentifier:
    """Treat dotted strings (com.apple.Finder) as bundle IDs; bare names as app names."""
    if "." in identifier and " " not in identifier:
        return AppIdentifier(bundle_id=identifier)
    return AppIdentifier(app_name=identifier)


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "recovery_suggestion": suggestion,
        },
    }


async def km_application_control(
    operation: Annotated[
        Literal["launch", "quit", "activate", "list_running", "get_state"],
        Field(description="Operation to perform on the application."),
    ],
    app_identifier: Annotated[
        str | None,
        Field(
            default=None,
            description="Application name (e.g. 'Safari') or bundle ID (e.g. 'com.apple.Safari'). "
            "Required for every operation except 'list_running'.",
            max_length=255,
        ),
    ] = None,
    force_quit: Annotated[
        bool,
        Field(
            default=False,
            description="When operation='quit', force-terminate instead of graceful quit. "
            "Force quit is blocked for security-sensitive apps.",
        ),
    ] = False,
    timeout_seconds: Annotated[
        float,
        Field(
            default=30.0,
            gt=0.0,
            le=300.0,
            description="Operation timeout in seconds (1-300).",
        ),
    ] = 30.0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Launch, quit, activate, or query macOS applications.

    Operations:
    - launch: start an app (or activate if already running)
    - quit: terminate gracefully (or forcibly if force_quit=True)
    - activate: bring a running app to the foreground
    - list_running: enumerate running applications (no app_identifier needed)
    - get_state: report whether an app is running/foreground/background/not_running

    Failure modes:
    - VALIDATION_ERROR: missing app_identifier where required, malformed bundle ID
    - SECURITY_BLOCKED: target app is on the security blacklist (Keychain, Terminal, etc.)
    - NOT_RUNNING: tried to activate an app that isn't running
    - APPLESCRIPT_FAILED: the underlying AppleScript call returned an error
    """
    if ctx:
        await ctx.info(f"km_application_control op={operation} target={app_identifier!r}")

    controller = _get_controller()

    if operation == "list_running":
        listing = await controller.get_running_applications_async()
        if listing.is_left():
            err = listing.get_left()
            return _failure(
                "APPLESCRIPT_FAILED",
                f"Failed to list running applications: {err.message}",
                "Ensure System Events accessibility permission is granted.",
            )
        return {"success": True, "data": {"running": listing.get_right()}}

    if not app_identifier or not app_identifier.strip():
        return _failure(
            "VALIDATION_ERROR",
            "app_identifier is required for this operation.",
            "Pass an application name like 'Safari' or a bundle ID like 'com.apple.Safari'.",
        )

    try:
        app_id = _build_app_id(app_identifier.strip())
    except ValueError as e:
        return _failure(
            "VALIDATION_ERROR",
            f"Invalid app identifier: {e}",
            "Use a plain app name or a dotted bundle ID (com.vendor.App).",
        )

    if operation == "launch":
        config = LaunchConfiguration(
            timeout=Duration.from_seconds(timeout_seconds),
        )
        result = await controller.launch_application_async(app_id, config)
    elif operation == "quit":
        result = await controller.quit_application_async(
            app_id,
            force=force_quit,
            timeout=Duration.from_seconds(timeout_seconds),
        )
    elif operation == "activate":
        result = await controller.activate_application(app_id)
    else:  # operation == "get_state" — Literal exhaustive
        state_result = await controller.get_application_state(app_id)
        if state_result.is_left():
            err = state_result.get_left()
            return _failure(
                "APPLESCRIPT_FAILED",
                f"Failed to query state: {err.message}",
                "Verify accessibility permissions and that the app identifier is correct.",
            )
        return {
            "success": True,
            "data": {
                "app": app_identifier,
                "state": state_result.get_right().value,
            },
        }

    if result.is_left():
        err = result.get_left()
        return _failure(
            "OPERATION_FAILED",
            err.message,
            "Check the app name/bundle ID and that the app is allowed by security policy.",
        )

    op_result = result.get_right()
    return {
        "success": op_result.success,
        "data": {
            "app": app_identifier,
            "operation": operation,
            "state": op_result.app_state.value,
            "duration_seconds": op_result.operation_time.total_seconds(),
            "details": op_result.details,
        },
    }
