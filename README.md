# Llama Guard Improvement via Output-Feedback Training

This project explores a method to improve LLM safeguards, specifically Meta's Llama Guard, by incorporating a learning mechanism. Standard safeguards often act as simple input/output gates (block/allow). This project investigates whether Llama Guard's *input* filtering capabilities can be enhanced by training it on prompts that previously led to blocked *outputs*.

![diagram](https://github.com/user-attachments/assets/ee62b312-d69e-440c-87fb-b5d66fd9cf6e)

The core idea: If a prompt is initially deemed safe by Llama Guard but the resulting LLM response is blocked by Llama Guard, that initial prompt can be used as negative training data. This aims to teach the input filter to identify and block potentially problematic prompts earlier, preventing the generation of harmful content downstream.

This repository contains the implementation (`research_implementation.py` ) and configuration (`research_config.json` ) for running experiments to test this hypothesis, using the Llama Guard finetuning techniques adapted from the official Llama cookbooks (`finetuning_data_formatter.py` ).

## How to Use

### Prerequisites

1.  **Python Environment:** Ensure you have Python 3 installed.
2.  **Dependencies:** Install the required packages:
    ```bash
    pip install -r requirements.txt 
    ```
3.  **Hugging Face Account & Access:**
    * You need a Hugging Face account.
    * Log in using the CLI: `huggingface-cli login`
    * Request access and accept the license terms for the models used (e.g., `meta-llama/LlamaGuard-7b`, `mistralai/Mistral-7B-Instruct-v0.3` ) on the Hugging Face website. The script checks for access before running.

### Configuration

* Modify `research_config.json`  to set paths, model names, dataset details, and training/evaluation parameters.
    * `dataset`: Specifies the dataset used for generating interactions and evaluation (e.g., `lmsys/toxic-chat` ).
    * `llama_guard`: Paths for the default and base Llama Guard models.
    * `llama`: Path for the base LLM used to generate responses (e.g., Mistral-7B).
    * `training`: Hyperparameters for fine-tuning the improved Llama Guard model.
    * `evaluation`: Configuration for evaluating model performance.
    * `experiment`: Settings for the full experiment run, like sample sizes.

### Running the Experiment

Use the `run_experiment.sh` script  to execute different stages of the research:

1.  **Process Dataset:** Generate interactions using the base LLM and filter them with the default Llama Guard to collect prompts that lead to blocked outputs.
    ```bash
    ./run_experiment.sh process
    ```
    This saves the problematic prompts to `./data/blocked_inputs/blocked_inputs.jsonl`.

2.  **Train Improved Model:** Fine-tune a Llama Guard model using the collected problematic prompts (and potentially safe examples, depending on the implementation) using LoRA.
    ```bash
    ./run_experiment.sh train
    ```
    The fine-tuned model adapter will be saved in the directory specified by `training.output_dir` in the config. The script updates the `evaluation.improved_model_path` in the config accordingly.

3.  **Evaluate Models:** Compare the performance (accuracy, precision, recall, F1-score) of the default Llama Guard and the fine-tuned ("improved") version on a test set.
    ```bash
    ./run_experiment.sh evaluate
    ```
    Results are saved in the directory specified by `evaluation.results_dir`.

4.  **Full Experiment:** Run all steps sequentially (process -> train -> evaluate).
    ```bash
    ./run_experiment.sh full
    ```
    Logs are saved to `experiment_*.log`.

## Learnings from Experiments

Initial experiments attempting to fine-tune Llama Guard using this output-feedback approach yielded poor results for the "improved" model. Key observations include:

* **Performance Degradation:** The fine-tuned model showed extremely low precision (~0.07) and accuracy (~0.08), although recall was perfect (1.0).
* **Bias Towards "Unsafe":** The metrics strongly suggest the fine-tuned model became overly cautious, classifying almost every input as "unsafe".
* **Likely Cause:** The poor performance likely stems from a mismatch between the training data/objective and Llama Guard's intended function. Fine-tuning solely on prompts labeled "unsafe" based on the *potential future output* (which isn't present during the input check) seems to teach the model spurious correlations rather than robustly identifying inherently unsafe prompts according to the defined categories. Training the model on this specific subset of data appears to create a strong bias, hindering its ability to correctly classify safe inputs.

Further experimentation is needed. Potential next steps include:
* Reviewing the code for any faults in implementation or logic errors in the code
* Adjusting the balance and selection of safe vs. unsafe examples during fine-tuning.
* Analyzing the characteristics of the prompts collected in `blocked_inputs.jsonl` to better understand why they lead to unsafe outputs.
* Exploring different training objectives or architectures instead of directly fine-tuning Llama Guard for this predictive task.
