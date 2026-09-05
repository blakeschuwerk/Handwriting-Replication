#!/bin/zsh
# Double-click this file in Finder to launch the Handwriting Studio dashboard.
# It activates the project venv, starts the local server, and opens your browser.
# Close the Terminal window (or press Ctrl-C) to stop it.

PROJECT="/Users/blakey5aces/Handwriting Analysis"
PORT="${HW_DASH_PORT:-8765}"
cd "$PROJECT" || { echo "Project folder not found"; exit 1; }

echo "Starting Handwriting Studio…"
# Open the browser shortly after the server comes up.
( sleep 1.5; open "http://127.0.0.1:${PORT}" ) &

exec "$PROJECT/.venv/bin/python3" "$PROJECT/dashboard/server.py"
