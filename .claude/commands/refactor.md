Refactor the specified code while preserving all existing behavior.

1. Read the target code thoroughly first
2. Identify: duplication, long functions, unclear naming, tight coupling
3. Apply refactoring in small, verified steps:
   - Extract method/function
   - Rename for clarity
   - Remove dead code
   - Simplify conditionals
4. After EACH step, run `pytest` to verify nothing broke
5. Run `ruff check src/` to ensure lint compliance

Do NOT change behavior. Do NOT add features. Refactor only.

$ARGUMENTS
