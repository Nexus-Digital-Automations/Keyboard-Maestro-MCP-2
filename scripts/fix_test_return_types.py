"""Add ``-> None`` to test functions missing a return annotation.

Walks ``tests/`` and rewrites every ``FunctionDef`` / ``AsyncFunctionDef``
whose ``returns`` is unset, **except** ones that contain a ``yield``
statement at any depth (those are generators and need a richer
annotation that this script doesn't infer).

Why not just run ``ruff --fix`` or ``autoflake``: neither adds missing
return annotations for ``no-untyped-def``; only libcst-level rewriting
gets the AST mutation right while preserving whitespace exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import libcst as cst


class ReturnsValue(cst.CSTVisitor):
    """True if the function body has a ``return <expr>`` or a ``yield``.

    Stops at nested function/lambda scopes so their returns don't leak up.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Yield(self, node: cst.Yield) -> None:
        self.found = True

    def visit_Return(self, node: cst.Return) -> None:
        if node.value is None:
            return
        if isinstance(node.value, cst.Name) and node.value.value == "None":
            return
        self.found = True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        return False


class AddNoneReturn(cst.CSTTransformer):
    """Annotate ``returns`` with ``None`` when missing on non-generator defs."""

    def __init__(self) -> None:
        self.changed = 0

    def _wants_none(self, node: cst.FunctionDef) -> bool:
        if node.returns is not None:
            return False
        visitor = ReturnsValue()
        node.body.visit(visitor)
        return not visitor.found

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if not self._wants_none(original_node):
            return updated_node
        self.changed += 1
        return updated_node.with_changes(
            returns=cst.Annotation(annotation=cst.Name("None"))
        )


def rewrite(path: Path) -> int:
    source = path.read_text()
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        print(f"  SKIP {path}: parse error ({exc})", file=sys.stderr)
        return 0
    transformer = AddNoneReturn()
    new_module = module.visit(transformer)
    if transformer.changed:
        path.write_text(new_module.code)
    return transformer.changed


def main(roots: list[str]) -> int:
    total = 0
    for root in roots:
        for py in Path(root).rglob("*.py"):
            n = rewrite(py)
            if n:
                print(f"{py}: +{n}")
                total += n
    print(f"\nTotal annotations added: {total}")
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or ["tests"]
    raise SystemExit(main(targets))
