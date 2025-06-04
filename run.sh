#!/bin/bash

# Load environment
source /net/tscratch/people/plgszymonjank/zzsn/.venv/bin/activate

# Run your Python script
mkdir logs -p
python3 -m main
