#!/usr/bin/env bash

set -e

# ============================================================
# Interactive Git Reword
#
# Usage:
#   ./git-reword.sh
#   ./git-reword.sh 10
#
# Controls:
#   ↑ / ↓   Move
#   Space   Toggle pick/reword
#   a       Select all as reword
#   n       Select none
#   m       Show commit details and diff
#   Enter   Confirm
#   q       Quit
# ============================================================

RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"
BLUE="\033[34m"

TMP_EDITOR=""

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

cleanup() {
    tput cnorm 2>/dev/null || true
    stty echo 2>/dev/null || true

    if [[ -n "$TMP_EDITOR" && -f "$TMP_EDITOR" ]]; then
        rm -f "$TMP_EDITOR"
    fi
}

trap cleanup EXIT INT TERM

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

die() {
    echo -e "${RED}Error:${RESET} $*" >&2
    exit 1
}

count_selected() {
    local COUNT=0

    for ((i = 0; i < N; i++)); do
        if (( SELECTED[i] == 1 )); then
            ((COUNT+=1))
        fi
    done

    echo "$COUNT"
}

clear_screen() {
    printf "\033[2J"
    printf "\033[H"
}

hide_cursor() {
    tput civis 2>/dev/null || true
}

show_cursor() {
    tput cnorm 2>/dev/null || true
}

# ------------------------------------------------------------
# Check Git repository
# ------------------------------------------------------------

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    die "Not inside a Git repository."
fi

if [[ ! -t 0 ]]; then
    die "This script requires an interactive terminal."
fi

# ------------------------------------------------------------
# Get number of commits
# ------------------------------------------------------------

N="${1:-}"

if [[ -z "$N" ]]; then
    read -r -p "How many commits do you want to inspect? " N
fi

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    die "Please enter a positive integer."
fi

COMMIT_COUNT=$(git rev-list --count HEAD)

if (( N > COMMIT_COUNT )); then
    die "Repository only has ${COMMIT_COUNT} commits."
fi

# ------------------------------------------------------------
# Check working tree
# ------------------------------------------------------------

if ! git diff --quiet || ! git diff --cached --quiet; then

    echo
    echo -e "${YELLOW}Warning:${RESET} You have uncommitted changes."
    echo

    git status --short

    echo
    read -r -p "Continue anyway? [y/N]: " ANSWER

    if [[ ! "$ANSWER" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ------------------------------------------------------------
# Get commits
#
# Same order as `git rebase -i`
# oldest -> newest
# ------------------------------------------------------------

mapfile -t COMMITS < <(
    git log \
        --reverse \
        --format='%H%x09%h%x09%s' \
        "HEAD~${N}..HEAD"
)

if (( ${#COMMITS[@]} != N )); then
    die "Failed to retrieve commits."
fi

# ------------------------------------------------------------
# Initialize selection
# ------------------------------------------------------------

declare -a SELECTED

for ((i = 0; i < N; i++)); do
    SELECTED[$i]=0
done

CURRENT=0

# ------------------------------------------------------------
# Draw menu
# ------------------------------------------------------------

draw_menu() {

    clear_screen

    echo -e "${BOLD}Interactive Git Reword${RESET}"
    echo
    echo -e "Rebase: ${CYAN}HEAD~${N}${RESET}"
    echo

    echo -e "${DIM}↑/↓ Move   Space Toggle   a All   n None   m Details   Enter Confirm   q Quit${RESET}"
    echo

    printf " %-2s %-10s %-9s %s\n" \
        "" "Hash" "Action" "Commit message"

    printf " %-2s %-10s %-9s %s\n" \
        "" "----------" "---------" "------------------------------"

    for ((i = 0; i < N; i++)); do

        IFS=$'\t' read -r FULL_HASH SHORT_HASH MESSAGE <<< "${COMMITS[$i]}"

        # Cursor
        if (( i == CURRENT )); then
            PREFIX="${CYAN}❯${RESET}"
        else
            PREFIX=" "
        fi

        # Action
        if (( SELECTED[i] == 1 )); then
            ACTION="${GREEN}reword${RESET}"
        else
            ACTION="${DIM}pick${RESET}"
        fi

        # Highlight current line
        if (( i == CURRENT )); then
            MESSAGE_DISPLAY="${BOLD}${MESSAGE}${RESET}"
        else
            MESSAGE_DISPLAY="$MESSAGE"
        fi

        printf " %b %-10s %-18b %s\n" \
            "$PREFIX" \
            "$SHORT_HASH" \
            "$ACTION" \
            "$MESSAGE_DISPLAY"
    done

    echo
    echo -e "${DIM}Selected: $(count_selected) / ${N}${RESET}"
}

# ------------------------------------------------------------
# Show commit details
# ------------------------------------------------------------

show_commit_details() {

    local INDEX="$CURRENT"

    IFS=$'\t' read -r FULL_HASH SHORT_HASH MESSAGE <<< "${COMMITS[$INDEX]}"

    clear_screen
    show_cursor

    echo -e "${BOLD}${CYAN}Commit Details${RESET}"
    echo
    echo -e "${BOLD}Commit:${RESET} ${FULL_HASH}"
    echo -e "${BOLD}Short:${RESET}  ${SHORT_HASH}"
    echo

    echo -e "${BOLD}Author:${RESET}"
    git show -s --format='%an <%ae>' "$FULL_HASH"

    echo
    echo -e "${BOLD}Date:${RESET}"
    git show -s --format='%ad' --date=iso "$FULL_HASH"

    echo
    echo -e "${BOLD}Subject:${RESET}"
    git show -s --format='%s' "$FULL_HASH"

    echo
    echo -e "${BOLD}Full Commit Message:${RESET}"
    echo "----------------------------------------"

    git show \
        -s \
        --format='%B' \
        "$FULL_HASH"

    echo "----------------------------------------"

    echo
    echo -e "${BOLD}Changed Files:${RESET}"
    echo

    git show \
        --format="" \
        --name-status \
        "$FULL_HASH"

    echo
    echo -e "${BOLD}Diff Stat:${RESET}"
    echo

    git show \
        --stat \
        --oneline \
        "$FULL_HASH"

    echo
    echo -e "${DIM}Press any key to return to the menu...${RESET}"

    IFS= read -rsn1
}

# ------------------------------------------------------------
# Read key
# ------------------------------------------------------------

read_key() {

    local KEY

    IFS= read -rsn1 KEY

    # ESC sequence
    if [[ "$KEY" == $'\x1b' ]]; then

        IFS= read -rsn2 KEY

        case "$KEY" in
            "[A")
                echo "UP"
                ;;
            "[B")
                echo "DOWN"
                ;;
            *)
                echo "ESC"
                ;;
        esac

        return
    fi

    case "$KEY" in

        " ")
            echo "SPACE"
            ;;

        "a"|"A")
            echo "ALL"
            ;;

        "n"|"N")
            echo "NONE"
            ;;

        "m"|"M")
            echo "DETAILS"
            ;;

        "q"|"Q")
            echo "QUIT"
            ;;

        "")
            echo "ENTER"
            ;;

        *)
            echo "OTHER"
            ;;

    esac
}

# ------------------------------------------------------------
# Interactive menu
# ------------------------------------------------------------

hide_cursor
stty -echo

while true; do

    draw_menu

    KEY=$(read_key)

    case "$KEY" in

        UP)

            if (( CURRENT > 0 )); then
                ((CURRENT-=1))
            fi

            ;;

        DOWN)

            if (( CURRENT < N - 1 )); then
                ((CURRENT+=1))
            fi

            ;;

        SPACE)

            if (( SELECTED[CURRENT] == 1 )); then
                SELECTED[CURRENT]=0
            else
                SELECTED[CURRENT]=1
            fi

            ;;

        ALL)

            for ((i = 0; i < N; i++)); do
                SELECTED[$i]=1
            done

            ;;

        NONE)

            for ((i = 0; i < N; i++)); do
                SELECTED[$i]=0
            done

            ;;

        DETAILS)

            show_commit_details

            ;;

        ENTER)

            break

            ;;

        QUIT)

            clear_screen
            show_cursor
            stty echo

            echo "Aborted."
            exit 0

            ;;

    esac

done

# ------------------------------------------------------------
# Check selection
# ------------------------------------------------------------

SELECTED_COUNT=$(count_selected)

if (( SELECTED_COUNT == 0 )); then

    clear_screen
    show_cursor
    stty echo

    echo -e "${YELLOW}No commits selected.${RESET}"
    echo "Nothing to reword."

    exit 0
fi

# ------------------------------------------------------------
# Confirmation screen
# ------------------------------------------------------------

clear_screen

echo -e "${BOLD}Selected commits to reword:${RESET}"
echo

REWORD_POSITIONS=""

for ((i = 0; i < N; i++)); do

    if (( SELECTED[i] == 1 )); then

        IFS=$'\t' read -r FULL_HASH SHORT_HASH MESSAGE <<< "${COMMITS[$i]}"

        echo -e "  ${GREEN}reword${RESET} ${SHORT_HASH} ${MESSAGE}"

        REWORD_POSITIONS+="$((i + 1)) "
    fi

done

echo
echo -e "Total: ${GREEN}${SELECTED_COUNT}${RESET} commit(s)"
echo

show_cursor
stty echo

read -r -p "Start git rebase -i HEAD~${N}? [y/N]: " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ------------------------------------------------------------
# Create temporary sequence editor
# ------------------------------------------------------------

TMP_EDITOR=$(mktemp)

cat > "$TMP_EDITOR" <<'EOF'
#!/usr/bin/env bash

TODO_FILE="$1"

for POS in $REWORD_POSITIONS; do

    # Replace only the corresponding "pick" line.
    sed -i.bak "${POS}s/^pick /reword /" "$TODO_FILE"

done

rm -f "${TODO_FILE}.bak"
EOF

chmod +x "$TMP_EDITOR"

# ------------------------------------------------------------
# Start rebase
# ------------------------------------------------------------

echo
echo -e "${CYAN}Starting interactive rebase...${RESET}"
echo

set +e

REWORD_POSITIONS="$REWORD_POSITIONS" \
GIT_SEQUENCE_EDITOR="$TMP_EDITOR" \
git rebase -i "HEAD~${N}"

STATUS=$?

set -e

# ------------------------------------------------------------
# Result
# ------------------------------------------------------------

echo

if (( STATUS == 0 )); then

    echo -e "${GREEN}${BOLD}✓ Rebase completed successfully.${RESET}"

else

    echo -e "${YELLOW}${BOLD}Rebase stopped or failed.${RESET}"

    echo
    echo "If you are resolving a conflict:"
    echo
    echo "  git status"
    echo "  git add <files>"
    echo "  git rebase --continue"
    echo
    echo "To abort:"
    echo
    echo "  git rebase --abort"

fi

exit "$STATUS"
