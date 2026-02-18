#!/usr/bin/env bash
#
# Standalone secret scanner for the march-madness-forecaster repo.
# Can be run manually or by CI. Scans ALL tracked files (not just staged).
#
# Usage:
#   ./scripts/scan-secrets.sh              # scan entire repo
#   ./scripts/scan-secrets.sh file1 file2  # scan specific files
#
# Exit codes:
#   0 = clean
#   1 = secrets found

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [ "$#" -gt 0 ]; then
    FILES="$*"
else
    FILES=$(git ls-files --cached -- "$REPO_ROOT" 2>/dev/null || find "$REPO_ROOT" -type f -not -path '*/.git/*')
fi

ERRORS=0

# ── Pattern definitions ──────────────────────────────────────────────
PATTERNS=(
    "AWS Access Key|AKIA[0-9A-Z]{16}"
    "AWS Secret Key|aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}"
    "GitHub Token|gh[ps]_[A-Za-z0-9_]{36,}"
    "GitHub OAuth|gho_[A-Za-z0-9_]{36,}"
    "OpenAI API Key|sk-[A-Za-z0-9]{20,}"
    "Generic API Key Assignment|[Aa][Pp][Ii][-_]?[Kk][Ee][Yy]\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
    "Generic Secret Assignment|[Ss][Ee][Cc][Rr][Ee][Tt]\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
    "Generic Token Assignment|[Tt][Oo][Kk][Ee][Nn]\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
    "Password Assignment|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*[:=]\s*['\"][^'\"]{4,}['\"]"
    "Private Key|-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\sKEY"
    "Bearer Token Hardcoded|Bearer\s+[A-Za-z0-9_\-\.]{20,}"
    "Basic Auth Hardcoded|Basic\s+[A-Za-z0-9+/=]{20,}"
    "Database URI with Credentials|(mongodb|postgres|postgresql|mysql|redis)://[^:]+:[^@]+@"
    "Slack Token|xox[bpors]-[0-9a-zA-Z]{10,}"
    "Stripe Key|[sr]k_(live|test)_[0-9a-zA-Z]{24,}"
    "Twilio SID|AC[a-f0-9]{32}"
)

BLOCKED_EXTENSIONS=(
    "\.pem$"
    "\.key$"
    "\.p12$"
    "\.pfx$"
    "\.keystore$"
)

BLOCKED_FILENAMES=(
    "^\.env$"
    "^\.env\."
    "credentials\.json$"
    "secrets\.json$"
    "service-account.*\.json$"
    "id_rsa$"
    "id_ed25519$"
)

echo "Scanning for secrets..."

# ── Check blocked file names/extensions ──────────────────────────────
for file in $FILES; do
    basename=$(basename "$file")

    for pattern in "${BLOCKED_EXTENSIONS[@]}"; do
        if echo "$basename" | grep -qE "$pattern"; then
            echo -e "${RED}BLOCKED${NC}: $file — sensitive file extension (${basename})"
            ERRORS=$((ERRORS + 1))
        fi
    done

    for pattern in "${BLOCKED_FILENAMES[@]}"; do
        if echo "$basename" | grep -qE "$pattern"; then
            echo -e "${RED}BLOCKED${NC}: $file — sensitive filename (${basename})"
            ERRORS=$((ERRORS + 1))
        fi
    done
done

# ── Check file contents ─────────────────────────────────────────────
for file in $FILES; do
    # Skip binary files
    if file "$file" 2>/dev/null | grep -q "binary"; then
        continue
    fi

    # Skip secret-scanning infrastructure (contains pattern strings that
    # match their own detection rules — not actual secrets).
    case "$(basename "$file")" in
        scan-secrets.sh|pre-commit|install-hooks.sh|secrets-qaqc.yml)
            continue
            ;;
    esac

    [ -f "$file" ] || continue

    for entry in "${PATTERNS[@]}"; do
        label="${entry%%|*}"
        regex="${entry##*|}"

        MATCHES=$(grep -nE "$regex" "$file" 2>/dev/null || true)
        if [ -n "$MATCHES" ]; then
            FILTERED=$(echo "$MATCHES" | grep -vE '(os\.getenv|os\.environ|#.*example|#.*placeholder|#.*TODO|export\s+\w+=\.\.\.|=\s*["\x27]?\.\.\.["\x27]?|scan-secrets|pre-commit)' || true)
            if [ -n "$FILTERED" ]; then
                echo -e "${RED}SECRET DETECTED${NC} [$label] in ${YELLOW}${file}${NC}:"
                echo "$FILTERED" | head -3 | while IFS= read -r line; do
                    echo "  $line"
                done
                MATCH_COUNT=$(echo "$FILTERED" | wc -l | tr -d ' ')
                if [ "$MATCH_COUNT" -gt 3 ]; then
                    echo "  ... and $((MATCH_COUNT - 3)) more matches"
                fi
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done
done

# ── Report ───────────────────────────────────────────────────────────
echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}FAIL: ${ERRORS} potential secret(s) found.${NC}"
    exit 1
else
    echo -e "${GREEN}PASS: No secrets detected.${NC}"
    exit 0
fi
