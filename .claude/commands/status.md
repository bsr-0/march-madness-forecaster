Analyze the current state of the project and working directory.

1. Run `git status` to show branch, staged/unstaged changes, untracked files
2. Run `git log --oneline -10` to show recent commits
3. Run `pytest --co -q 2>/dev/null | tail -5` to count tests
4. Run `ruff check src/ --quiet 2>&1 | wc -l` to count lint issues
5. Summarize: branch, uncommitted changes, test count, lint status, last commit
