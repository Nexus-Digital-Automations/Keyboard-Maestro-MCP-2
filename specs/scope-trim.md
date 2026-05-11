# Spec: Scope Trim to Keyboard-Maestro-only Surface

Approved plan: `/Users/jeremyparker/.claude/plans/i-want-this-mcp-logical-cloud.md`

## Acceptance criteria

1. `python -c "import src.main_dynamic"` succeeds with zero `ModuleNotFoundError`.
2. Server startup enumerates ≤ 35 MCP tools, all `km_*`, none matching `km_predict_*`, `km_iot_*`, `km_voice_*`, `km_autonomous_*`, `km_detect_*`, `km_classify_*`, `km_ai_*`, `km_nlp_*`, `km_smart_*`, `km_forecast_*`.
3. Repo grep for `applescript|AppleScript|km_engine|KMClient|km_client|osascript` is positive for every retained tool file in `src/server/tools/` (or its direct dependency chain).
4. No `src/server/tools/*.py` retained file imports a deleted `src/` package (`src.ai`, `src.iot`, `src.agents`, `src.voice`, `src.vision`, `src.nlp`, `src.prediction`, `src.intelligence`, `src.analytics`, `src.suggestions`, `src.orchestration`, `src.tokens`).
5. `pyproject.toml` no longer declares OpenAI, speech-recognition, computer-vision, IoT-protocol, or scikit-learn-only-for-prediction dependencies; `pip install -e .` succeeds in a clean resolve.
6. `pytest tests/` runs to completion (failures permitted only if they expose bugs in retained code that pre-dated this trim — never new ones).
7. README sections titled "AI Intelligence", "IoT Integration", "Voice Control", "Predictive Analytics", "Computer Vision", "Natural Language", "Autonomous Agents" are gone; tool list reflects actual surface.
8. Each wave (A/B/C, src-subtrees, core-architecture, deps, tests, README) lands as its own commit so any wave is independently revertible.

## Out of scope

- Refactoring retained tools' internals.
- Adding new KM-bridge functionality.
- Touching `src/integration/km_client.py` (the actual KM bridge — sacrosanct).
