#!/bin/bash

# ---------------------------
# GSM/RTL-SDR ML Setup Script
# ---------------------------

# Name of the virtual environment
VENV_NAME="rtl_ml_chest_env"

# Path to your script
SCRIPT_PATH="/home/wiguy/Documents/rtl_sdr_ml_channel_prediction_classic_ml_realtime.py"

# Create virtual environment
python3 -m venv "$VENV_NAME"

# Activate virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip
#pip install --upgrade pip
#pip install pyrtlsdr
# Install compatible packages
#pip install numpy scipy scikit-learn matplotlib

# Run the RTL-SDR ML script
python3 "$SCRIPT_PATH" --fc 98.3e6 --samp-rate 1e6 --gain 40.2 --frame 16384 --hop 8192 --horizon-ms 3 --win 16 --retrain-interval 1.0

# Deactivate virtual environment after running
deactivate

