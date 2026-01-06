#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

