"""No source file may use a module it never imported.

`src/pipeline/tournament_pipeline.py` called `math.exp` in the production
probability path (`_raw_fusion_probability`) with no `import math` -- lost in
the April 2026 cleanup, and unnoticed for five months because the pipeline
could not get far enough to reach the line. A NameError in the one function
that turns model output into a win probability is not something the test
suite should need a 12-minute end-to-end run to find.

Static check: for a handful of commonly used modules, any `name.attr` use in
a file that has no import binding `name` is a failure. Deliberately narrow --
this is not a linter, it is a tripwire for the exact failure mode above.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("math", "json", "os", "re", "time", "logging", "datetime", "itertools", "collections", "np", "pd")
FILES = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)


def _bound_names(tree: ast.AST) -> set:
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    return bound


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_module_attribute_uses_are_imported(path):
    tree = ast.parse(path.read_text())
    bound = _bound_names(tree)
    offenders = sorted(
        {
            f"{node.value.id}.{node.attr} @ line {node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in MODULES
            and node.value.id not in bound
        }
    )
    assert not offenders, f"uses a module never imported in this file: {offenders[:5]}"
