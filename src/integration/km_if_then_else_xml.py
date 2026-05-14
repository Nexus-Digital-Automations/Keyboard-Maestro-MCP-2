"""Emit KM-canonical IfThenElse action plist XML.

@stable Shared by km_add_condition and km_control_flow (if_then_else
mode). The XML matches Keyboard Maestro 11's "Copy as XML" output for
the IfThenElse action — verified against captured templates in
src/server/data/km_action_templates.json (key "IfThenElse").

Decision record: a previous emitter in src/integration/km_control_flow.py
produced custom XML (``<action id="..." name="If Then Else">`` with
``<then>``/``<else>`` children) that KM never accepted. That code is
unused dead weight; this module replaces it for the if_then_else path.

Failure modes:
- UnsupportedConditionType: condition_type not in {variable, text,
  application, calculation}.
- UnsupportedOperator: operator not mappable for the given condition_type.
"""

from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape_lib

# Operator → KM ConditionType-specific keyword.
# KM uses Is/IsNot/Contains/DoesNotContain etc. depending on category.
_VARIABLE_OPERATOR_MAP = {
    "equals": "Is",
    "not_equals": "IsNot",
    "contains": "Contains",
    "not_contains": "DoesNotContain",
    "greater": "NumericallyGreaterThan",
    "less": "NumericallyLessThan",
    "regex": "MatchesRegex",
    "exists": "IsNotEmpty",
    "is_empty": "IsEmpty",
    "is_not_empty": "IsNotEmpty",
}

_TEXT_OPERATOR_MAP = {
    "equals": "Is",
    "not_equals": "IsNot",
    "contains": "Contains",
    "not_contains": "DoesNotContain",
    "regex": "MatchesRegex",
}

_APPLICATION_OPERATOR_MAP = {
    "exists": "Running",
    "equals": "Active",  # "is the active app"
    "not_equals": "NotActive",
}


class UnsupportedConditionType(ValueError):
    """Raised when condition_type cannot be mapped to a KM ConditionType."""


class UnsupportedOperator(ValueError):
    """Raised when operator cannot be mapped for the chosen condition_type."""


def escape(text: str) -> str:
    """XML-escape a value for inclusion in a KM plist string body."""
    return _xml_escape_lib(text, {'"': "&quot;", "'": "&apos;"})


def build_condition_dict(
    condition_type: str,
    operator: str,
    operand: str,
    *,
    case_sensitive: bool = True,
    negate: bool = False,
) -> str:
    """Return one KM ConditionList <dict> entry as XML.

    The returned string is a complete ``<dict>...</dict>`` plist fragment
    suitable for inclusion in a ConditionList ``<array>``.
    """
    kind = condition_type.lower().strip()
    if kind == "variable":
        return _variable_condition(operator, operand, negate=negate)
    if kind == "text":
        return _text_condition(operator, operand, case_sensitive=case_sensitive, negate=negate)
    if kind == "application":
        return _application_condition(operator, operand, negate=negate)
    if kind == "calculation":
        return _calculation_condition(operand, negate=negate)
    raise UnsupportedConditionType(
        f"condition_type {condition_type!r} not supported. "
        "Supported: variable, text, application, calculation.",
    )


def build_if_then_else_xml(
    condition_xml: str,
    then_actions_xml: str = "",
    else_actions_xml: str = "",
    *,
    match: str = "All",
    timeout_aborts: bool = True,
) -> str:
    """Wrap a condition <dict> + inner action <dict>s into a full IfThenElse plist.

    ``then_actions_xml`` / ``else_actions_xml`` are pre-rendered action
    ``<dict>`` strings concatenated with no wrapper (KM expects them as
    ``<array>`` items, so each must be a top-level ``<dict>``).
    """
    return (
        "<dict>"
        "<key>Conditions</key>"
        "<dict>"
        "<key>ConditionList</key>"
        f"<array>{condition_xml}</array>"
        "<key>ConditionListMatch</key>"
        f"<string>{escape(match)}</string>"
        "</dict>"
        "<key>ElseActions</key>"
        f"<array>{else_actions_xml}</array>"
        "<key>MacroActionType</key>"
        "<string>IfThenElse</string>"
        "<key>ThenActions</key>"
        f"<array>{then_actions_xml}</array>"
        "<key>TimeOutAbortsMacro</key>"
        f"<{'true' if timeout_aborts else 'false'}/>"
        "</dict>"
    )


def _variable_condition(operator: str, operand: str, *, negate: bool) -> str:
    op = _VARIABLE_OPERATOR_MAP.get(operator.lower())
    if op is None:
        raise UnsupportedOperator(
            f"operator {operator!r} not supported for variable conditions. "
            f"Supported: {sorted(_VARIABLE_OPERATOR_MAP)}.",
        )
    if negate:
        op = _negate_operator(op)
    parts = [
        "<key>ConditionType</key><string>Variable</string>",
        f"<key>VariableConditionType</key><string>{escape(op)}</string>",
        f"<key>Variable</key><string>{escape(operand_to_var(operand))}</string>",
    ]
    if op in {"Is", "IsNot", "Contains", "DoesNotContain", "MatchesRegex",
              "NumericallyGreaterThan", "NumericallyLessThan"}:
        parts.append(
            f"<key>ConditionResult</key><string>{escape(operand_after_first(operand))}</string>",
        )
    return "<dict>" + "".join(parts) + "</dict>"


def _text_condition(
    operator: str,
    operand: str,
    *,
    case_sensitive: bool,
    negate: bool,
) -> str:
    op = _TEXT_OPERATOR_MAP.get(operator.lower())
    if op is None:
        raise UnsupportedOperator(
            f"operator {operator!r} not supported for text conditions. "
            f"Supported: {sorted(_TEXT_OPERATOR_MAP)}.",
        )
    if negate:
        op = _negate_operator(op)
    var_name, compare = _split_text_operand(operand)
    case_key = "" if case_sensitive else "<key>CaseSensitive</key><false/>"
    return (
        "<dict>"
        "<key>ConditionType</key><string>TextContents</string>"
        f"<key>TextContentsConditionType</key><string>{escape(op)}</string>"
        f"<key>Text</key><string>{escape(var_name)}</string>"
        f"<key>ConditionResult</key><string>{escape(compare)}</string>"
        f"{case_key}"
        "</dict>"
    )


def _application_condition(operator: str, operand: str, *, negate: bool) -> str:
    op = _APPLICATION_OPERATOR_MAP.get(operator.lower())
    if op is None:
        raise UnsupportedOperator(
            f"operator {operator!r} not supported for application conditions. "
            f"Supported: {sorted(_APPLICATION_OPERATOR_MAP)}.",
        )
    if negate:
        op = "Not" + op if not op.startswith("Not") else op[3:]
    return (
        "<dict>"
        "<key>ConditionType</key><string>Application</string>"
        f"<key>ApplicationConditionType</key><string>{escape(op)}</string>"
        "<key>Application</key>"
        "<dict>"
        f"<key>BundleIdentifier</key><string>{escape(operand)}</string>"
        f"<key>Name</key><string>{escape(operand)}</string>"
        "</dict>"
        "</dict>"
    )


def _calculation_condition(operand: str, *, negate: bool) -> str:
    kind = "CalculationIsFalse" if negate else "CalculationIsTrue"
    return (
        "<dict>"
        "<key>ConditionType</key><string>Calculation</string>"
        f"<key>CalculationConditionType</key><string>{escape(kind)}</string>"
        f"<key>CalculationConditionExpression</key><string>{escape(operand)}</string>"
        "</dict>"
    )


def operand_to_var(operand: str) -> str:
    """Return the variable name portion of a 'VarName=value' operand.

    KM variable conditions split the operand into Variable + ConditionResult.
    Callers may pass either ``"VarName"`` (Is/IsNot/Contains comparing to
    empty) or ``"VarName=value"``.
    """
    if "=" in operand:
        return operand.split("=", 1)[0].strip()
    return operand.strip()


def operand_after_first(operand: str) -> str:
    """Return the comparison-value portion of a 'VarName=value' operand."""
    if "=" in operand:
        return operand.split("=", 1)[1]
    return operand


def _split_text_operand(operand: str) -> tuple[str, str]:
    """Split ``"left_text=right_text"`` into (left, right) for TextContents."""
    if "=" in operand:
        left, right = operand.split("=", 1)
        return left.strip(), right
    return operand, ""


def _negate_operator(op: str) -> str:
    swaps = {
        "Is": "IsNot",
        "IsNot": "Is",
        "Contains": "DoesNotContain",
        "DoesNotContain": "Contains",
        "NumericallyGreaterThan": "NumericallyLessThanOrEqual",
        "NumericallyLessThan": "NumericallyGreaterThanOrEqual",
        "MatchesRegex": "DoesNotMatchRegex",
        "IsEmpty": "IsNotEmpty",
        "IsNotEmpty": "IsEmpty",
    }
    return swaps.get(op, op)


def build_execute_macro_action(target_macro: str) -> str:
    """Return an ExecuteMacro <dict> targeting ``target_macro`` by name or UID."""
    return (
        "<dict>"
        "<key>MacroActionType</key><string>ExecuteMacro</string>"
        f"<key>Macro</key><string>{escape(target_macro)}</string>"
        "</dict>"
    )
