#!/bin/bash
# Run Faceboard Nova

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the GUI
python3 ar_face_masks_gui.py
