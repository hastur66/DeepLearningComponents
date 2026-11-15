#!/bin/bash
# Supported GPUs: NVIDIA Hopper (H100), Ampere (A100, A10)
# FP8 requires Hopper (H100) or Ada Lovelace; FP16 runs on Ampere and later
# this will fail on devices with a small shared memory size, such as RTX-4090

# FP8
python persistent_matmul.py --prec fp8 --K_range 128 1024 --K_step 128

# FP16 (includes CLC variants on supported devices)
python persistent_matmul.py --prec fp16 --K_range 128 1024 --K_step 128
