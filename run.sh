#!/bin/bash

# Load environment
source /net/tscratch/people/plgrszczepaniak/.envs/zzsn/bin/activate

# Run your Python script
mkdir logs -p
python3 -m main
