"""Every entry in ``mc_pool_backtest``'s legacy mode registry must be a 3-tuple.

The specs are consumed by ``for mode_name, rp, sampler in mode_sampler_specs``,
so an entry of any other length raises ``ValueError: too many values to unpack``
the moment its mode is enabled -- deep inside a backtest run, long after the
typo.

This is not hypothetical.  ``lev_tilt_200`` shipped with 26 stray mode-name
strings pasted between its name and its base, duplicating the block that
already exists in ``ALL_MODES``.  It survived because the specs are built inside
``_run_one_year`` (so nothing imports them cheaply), because the only shape
check on the path looks at ``spec[1] is None`` -- and a stray string is not
None -- and because the entry was indented so that the strings read as though
they belonged to the enclosing dict rather than to the tuple.

The registry is read statically here for that first reason: building it for real
needs a year's worth of probability bases on disk, which is exactly the cost
that let the defect through.
"""

import ast
from pathlib import Path

import pytest

BACKTEST = Path(__file__).resolve().parents[1] / "scripts" / "mc_pool_backtest.py"


def _legacy_spec_dict() -> ast.Dict:
    """The ``legacy_specs = {...}`` literal from ``_run_one_year``."""
    tree = ast.parse(BACKTEST.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "legacy_specs" in targets and isinstance(node.value, ast.Dict):
            return node.value
    pytest.fail(f"no `legacy_specs = {{...}}` literal found in {BACKTEST}")


def _named_specs():
    literal = _legacy_spec_dict()
    for key, value in zip(literal.keys, literal.values):
        assert isinstance(key, ast.Constant), "mode keys are expected to be literals"
        yield key.value, value


def test_every_legacy_spec_is_a_name_base_sampler_triple():
    wrong = {
        name: len(value.elts)
        for name, value in _named_specs()
        if isinstance(value, ast.Tuple) and len(value.elts) != 3
    }
    assert not wrong, (
        "legacy mode specs must be (mode_name, base, sampler); these are not: "
        f"{wrong}. Unpacking at `for mode_name, rp, sampler in mode_sampler_specs` "
        "will raise as soon as the mode is enabled."
    )


def test_every_legacy_spec_names_itself_first():
    """The first element is the mode's own name; a paste error shows up here too."""
    mismatched = {
        name: value.elts[0].value
        for name, value in _named_specs()
        if isinstance(value, ast.Tuple)
        and value.elts
        and isinstance(value.elts[0], ast.Constant)
        and value.elts[0].value != name
    }
    assert not mismatched, f"spec name does not match its registry key: {mismatched}"
