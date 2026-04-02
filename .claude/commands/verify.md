Verify the current work is actually complete using the verification-before-completion skill.

1. Run `pytest -v` — report exact pass/fail counts
2. Run `ruff check src/` — report exact error count
3. Check all modified files for: unused imports, dead code, missing tests
4. State what was verified with evidence — no "should work" or "looks good"
