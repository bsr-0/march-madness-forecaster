#!/usr/bin/env bash
#
# Install git hooks for the march-madness-forecaster repo.
# Run once after cloning:  ./scripts/install-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing pre-commit hook..."

cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/usr/bin/env bash
# Auto-installed pre-commit hook — runs secret scanner on staged files.
# To bypass: git commit --no-verify

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Run the full scanner on staged files only
exec "$REPO_ROOT/scripts/scan-secrets.sh" $STAGED_FILES
HOOK

chmod +x "$HOOKS_DIR/pre-commit"
echo "Pre-commit hook installed at $HOOKS_DIR/pre-commit"
echo "Done. Secrets will be scanned before every commit."
