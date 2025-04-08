#!/bin/bash
# Script to run the Llama Guard Improvement Research Experiment

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
CONFIG_PATH="research_config.json"
LOG_FILE="experiment_$(date +%Y%m%d_%H%M%S).log"
RESULTS_DIR="./results"

# Create directories
mkdir -p ${RESULTS_DIR}/experiment
mkdir -p ${RESULTS_DIR}/evaluation
mkdir -p ./data/dataset
mkdir -p ./data/blocked_inputs
mkdir -p ./models/improved

# Create log file
touch $LOG_FILE
exec > >(tee -a $LOG_FILE) 2>&1

echo "======================================"
echo "Llama Guard Improvement Research Experiment"
echo "Started at: $(date)"
echo "======================================"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file $CONFIG_PATH not found!"
    exit 1
fi

# Check for required Python packages
echo "Checking required packages..."
pip install -q transformers datasets torch bitsandbytes tqdm pandas scikit-learn accelerate peft sentencepiece protobuf

# Set the seed environment variable for extra reproducibility
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

# Check for access to model repositories
echo "Checking model access..."
python -c "
import json
from typing import Dict, Any
from huggingface_hub import model_info
with open('$CONFIG_PATH', 'r') as f:
    config = json.load(f)
llama_guard_model = config['llama_guard']['default_model_path']
llama_model = config['llama']['model_path']
try:
    model_info(llama_guard_model)
    print(f'✅ Access confirmed to {llama_guard_model}')
except Exception as e:
    print(f'❌ Cannot access {llama_guard_model}. Error: {e}')
    print('Please login with: huggingface-cli login')
    print('And ensure you have accepted the model license on the Hugging Face website')
    exit(1)
try:
    model_info(llama_model)
    print(f'✅ Access confirmed to {llama_model}')
except Exception as e:
    print(f'❌ Cannot access {llama_model}. Error: {e}')
    print('Please ensure you have accepted the model license on the Hugging Face website')
    exit(1)
" || {
    echo "⚠️  Warning: Could not access required models. Make sure you have:";
    echo "1. Logged in to Hugging Face with 'huggingface-cli login'";
    echo "2. Accepted the license agreement for the Meta Llama models";
    echo "3. Have proper internet connection";
    exit 1;
}

# Run the experiment
echo "Starting the experiment..."

if [ "$1" == "full" ]; then
    echo "Running full experiment pipeline"
    python research_implementation.py --config $CONFIG_PATH --full-experiment
elif [ "$1" == "process" ]; then
    echo "Processing dataset to collect blocked inputs"
    # Set CUDA_VISIBLE_DEVICES for GPU determinism if multiple GPUs are available
    CUDA_VISIBLE_DEVICES=0 python research_implementation.py --config $CONFIG_PATH --process-dataset
elif [ "$1" == "train" ]; then
    echo "Training improved Llama Guard model"
    CUDA_VISIBLE_DEVICES=0 python research_implementation.py --config $CONFIG_PATH --train
elif [ "$1" == "evaluate" ]; then
    echo "Evaluating default and improved models"
    CUDA_VISIBLE_DEVICES=0 python research_implementation.py --config $CONFIG_PATH --evaluate
else
    echo "Please specify a mode: full, process, train, or evaluate"
    echo "Example: ./run_experiment.sh full"
    exit 1
fi

echo "======================================"
echo "Experiment completed at: $(date)"
echo "Logs available at: $LOG_FILE"
echo "======================================"

# Print helpful information
echo "Experiment Results Summary:"
if [ -d "$RESULTS_DIR/experiment" ]; then
    LATEST_RESULT=$(ls -t $RESULTS_DIR/experiment/experiment_results_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo "Latest experiment results:"
        python -c "
import json
try:
    with open('$LATEST_RESULT', 'r') as f:
        result = json.load(f)
    print(json.dumps(result.get('evaluation', {}), indent=2))
except Exception as e:
    print(f'Error reading results: {e}')
"
    else
        echo "No experiment results found yet."
    fi
fi

echo "======================================"
echo "To analyze a specific query, run:"
echo "python research_implementation.py --config $CONFIG_PATH --query \"your query here\""
echo ""
echo "To view detailed results, check the files in $RESULTS_DIR"