#!/bin/bash

# Define the Python script to run
python_script="optimisation_2024.py"

# Define the arguments to pass to the Python script
data="/shared/scratch/wq22321/data/NANOAOD_Muon0Run2023D_ZMu_PromptReco_v2RAW_RECO_2023_v0_4/231121_100830/0000/out_*.root"
metric="turnon"
workers=-1

# Run the Python script with the defined arguments
python3 "$python_script" "$data" "$metric" "$workers"