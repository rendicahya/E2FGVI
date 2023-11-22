#!/bin/bash

GITMODULES_FILE=".gitmodules"

if [ -f "$GITMODULES_FILE" ]; then
    submodule_names=$(grep -E '^\[submodule "' "$GITMODULES_FILE" | sed 's/^\[submodule "//;s/"\]$//')

    for dir in $submodule_names; do
        if [ -d "$dir" ]; then
            (cd "$dir" && git pull)
        fi
    done
else
    echo "Error: .gitmodules file not found."
fi
