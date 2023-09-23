#!/usr/bin/env bash

colmap_path="/usr/local/bin"
base_path="/mnt/sda/datasets/MegaDepth"

python3 undistor_reconstructions.py --colmap_path="${colmap_path}" --base_path="${base_path}"