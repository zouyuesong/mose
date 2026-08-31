#!/bin/bash
# Run csvport/jobd with the vendored MOSEK 9.3 runtime libraries (lib/).
# The pip package "Mosek==9.3.22" ships the same libs; we vendor them so no
# full MOSEK installation is needed. A license is still required:
#   put your mosek.lic at ~/mosek/mosek.lic  (or export MOSEKLM_LICENSE_FILE)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x "$DIR/bin/jobd" ]; then
    exec "$DIR/bin/jobd" "$@"
else
    exec "$DIR/bin/csvport" "$@"
fi
