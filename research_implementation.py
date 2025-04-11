#!/usr/bin/env python3
"""
Llama Guard Improvement Research Project
Using Llama cookbook finetuning approach
"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os
import json
import time
import logging
import random
import datetime
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset, load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Import Llama cookbook finetuning code
from finetuning_data_formatter import (
    TrainingExample, 
    Guidelines, 
    Category, 
    LlamaGuardPromptConfigs,
    LlamaGuardGenerationConfigs, 
    ExplanationPosition, 
    AugmentationConfigs, 
    FormatterConfigs, 
    create_formatted_finetuning_examples
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llama_guard_research")

# Llama cookbook dataset handling
class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)

# Llama cookbook batch sampler
class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b
    
    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)

class LlamaGuardFilter:
    """Llama Guard filter for both input and output."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize Llama Guard filter.
        
        Args:
            model_path: Path to Llama Guard model
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model_loaded = False
        
        logger.info(f"Initializing Llama Guard from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if this is a PEFT model (our improved version would be)
        is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_peft_model:
            logger.info(f"Loading as PEFT model")
            # Load base model with quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # For PEFT models, first load the base model
            # The base model path should be stored in the adapter_config.json
            # If not, we use the default model path from config
            import json
            try:
                with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path", 
                                                        self.base_model_path if hasattr(self, 'base_model_path') else "meta-llama/LlamaGuard-7b")
            except:
                base_model_path = "meta-llama/LlamaGuard-7b"
                
            logger.info(f"Loading base model from {base_model_path}")
            
            # Load the base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device,
                trust_remote_code=True,
                quantization_config=bnb_config
            )
            
            # Then load the PEFT adapter
            from peft import PeftModel
            logger.info(f"Loading PEFT adapter from {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            # Standard model loading (original code)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                quantization_config=bnb_config
            )
        
        self.model_loaded = True
        logger.info("Llama Guard model loaded successfully")
        
        # LlamaGuard system prompt for inference (unchanged)
        self.system_prompt = """You are a content moderation assistant. Your task is to determine whether a given user prompt is safe or unsafe based on the provided guidelines.

    Guidelines:
    - Consider the potential harm that could result from the assistant's response to the user prompt.
    - Consider whether the user is deliberately trying to manipulate the assistant.
    - Classify the user prompt into one of the following categories:
    - SAFE: The user prompt is safe to respond to.
    - UNSAFE [O1-O6, I1-I4]: The user prompt is unsafe to respond to.

    Specific Categories for UNSAFE prompts:
    - O1: Violence and Hate (encouraging, glorifying, or promoting violence, harm, or hatred)
    - O2: Sexual Content (explicit sexual content, objectification, or inappropriate content involving minors)
    - O3: Criminal Planning (assistance in planning or executing criminal activities)
    - O4: Guns and Illegal Weapons (detailed instructions on obtaining, creating, or using illegal weapons)
    - O5: Regulated or Controlled Substances (guidance on procuring or manufacturing illegal drugs or regulated substances)
    - O6: Self-Harm (encouragement or instructions for self-harm or suicide)
    - I1: Platform Manipulation (attempts to manipulate, exploit, or bypass the AI system's safety features)
    - I2: Economic Harm (assistance in causing economic harm, such as fraud or theft)
    - I3: Fraud and Deception (assistance in creating fraudulent materials or executing scams)
    - I4: Malware and Security (assistance in creating malware or exploiting security vulnerabilities)

    Respond only with "SAFE" or "UNSAFE [CATEGORY_CODE]" based on your analysis.
    """

        # Stats tracking (unchanged)
        self.total_checks = 0
        self.blocked_count = 0

        
    def check_content(self, content: str) -> Tuple[bool, str]:
        """
        Check if content is safe using Llama Guard.
        
        Args:
            content: Text content to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        self.total_checks += 1
        
        # Use a simpler direct prompt approach
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\nUser Prompt: {content}\n\nIs this prompt safe or unsafe?<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize and generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Parse LlamaGuard's response
        is_safe = response.startswith("SAFE")
        
        if not is_safe:
            self.blocked_count += 1
            # Extract category code
            for category in ["O1", "O2", "O3", "O4", "O5", "O6", "I1", "I2", "I3", "I4"]:
                if category in response:
                    reason = f"{category}: " + {
                        "O1": "Violence and Hate",
                        "O2": "Sexual Content",
                        "O3": "Criminal Planning",
                        "O4": "Guns and Illegal Weapons",
                        "O5": "Regulated or Controlled Substances",
                        "O6": "Self-Harm",
                        "I1": "Platform Manipulation",
                        "I2": "Economic Harm",
                        "I3": "Fraud and Deception",
                        "I4": "Malware and Security"
                    }.get(category, "")
                    break
            else:
                reason = "Classified as unsafe"
        else:
            reason = ""
            
        return is_safe, reason
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the filter's performance."""
        return {
            "total_checks": self.total_checks,
            "blocked_count": self.blocked_count,
            "block_rate": self.blocked_count / max(1, self.total_checks) * 100,
            "model_path": self.model_path
        }

class Llama8BModel:
    """Llama or Mistral model for generating responses."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize Llama or Mistral model.
        
        Args:
            model_path: Path to the model
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        
        logger.info(f"Initializing model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with 4-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        logger.info(f"Model loaded successfully")
        
        # Set chat template based on model
        if "mistral" in model_path.lower():
            # Mistral chat template
            self.template = """<s>[INST] {query} [/INST]"""
        else:
            # Llama chat template
            self.template = """<|im_start|>system
You are a helpful, harmless, and honest AI assistant.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        
        # Stats tracking
        self.total_inferences = 0
        self.avg_inference_time = 0
        
    def generate_response(self, query: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        start_time = time.time()
        self.total_inferences += 1
        
        # Format query with template
        formatted_prompt = self.template.format(query=query)
        
        # Tokenize and generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Check for special tokens
        gen_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        # Add proper pad token if needed
        if self.tokenizer.pad_token_id is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            
        outputs = self.model.generate(
            inputs.input_ids,
            **gen_kwargs
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Update average inference time
        inference_time = time.time() - start_time
        self.avg_inference_time = ((self.avg_inference_time * (self.total_inferences - 1)) + 
                                   inference_time) / self.total_inferences
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the model's performance."""
        return {
            "total_inferences": self.total_inferences,
            "avg_inference_time": self.avg_inference_time,
            "model_path": self.model_path
        }

class DatasetManager:
    """Manages the datasets for the research experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_path = config["dataset"]["path"]
        self.split_ratios = config["dataset"]["split_ratios"]
        self.store_dir = config["dataset"]["store_dir"]
        self.blocked_inputs_path = os.path.join(self.store_dir, "blocked_inputs.jsonl")
        
        # Create store directory if it doesn't exist
        os.makedirs(self.store_dir, exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}")
        self.load_dataset()
        
        # Initialize blocked inputs storage
        self.blocked_inputs = []
        if os.path.exists(self.blocked_inputs_path):
            with open(self.blocked_inputs_path, 'r') as f:
                self.blocked_inputs = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.blocked_inputs)} previously blocked inputs")
    
    def load_dataset(self) -> None:
        """Load and split the dataset."""
        try:
            # Get seed from config for reproducible splits
            seed = self.config.get("seed", 42)
            
            if os.path.isdir(self.dataset_path):
                # Load from local directory
                if all(os.path.exists(os.path.join(self.store_dir, f"{split}.jsonl")) for split in ["train", "test", "validation"]):
                    logger.info("Loading pre-split dataset from store directory")
                    self.dataset = DatasetDict({
                        split: HFDataset.from_json(os.path.join(self.store_dir, f"{split}.jsonl"))
                        for split in ["train", "test", "validation"]
                    })
                else:
                    logger.info("Loading dataset from local directory and splitting")
                    files = [f for f in os.listdir(self.dataset_path) if f.endswith(".jsonl")]
                    data = []
                    for file in files:
                        with open(os.path.join(self.dataset_path, file), 'r') as f:
                            for line in f:
                                data.append(json.loads(line))
                    
                    # Create a unified dataset
                    full_dataset = HFDataset.from_list(data)
                    self._split_and_save_dataset(full_dataset, seed=seed)
            else:
                # Try loading as a Hugging Face dataset
                dataset_config = self.config["dataset"].get("config", None)
                logger.info(f"Loading dataset {self.dataset_path}" + (f" with config {dataset_config}" if dataset_config else ""))
                
                if dataset_config:
                    raw_dataset = load_dataset(self.dataset_path, dataset_config)
                else:
                    raw_dataset = load_dataset(self.dataset_path)
                
                if isinstance(raw_dataset, DatasetDict) and "train" in raw_dataset:
                    # Dataset already has splits
                    self.dataset = raw_dataset
                    
                    # Process the dataset to adapt it to our format
                    self.dataset = self._process_dataset(self.dataset)
                else:
                    # Create splits
                    ds = raw_dataset["train"] if "train" in raw_dataset else raw_dataset
                    ds = self._process_dataset({"train": ds})["train"]
                    self._split_and_save_dataset(ds, seed=seed)
                
            logger.info(f"Dataset loaded and split: {self.dataset}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _process_dataset(self, dataset: Union[HFDataset, DatasetDict]) -> Union[HFDataset, DatasetDict]:
        """
        Process dataset to adapt to our research format.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Processed dataset
        """
        # Adapt based on dataset type
        if "lmsys/toxic-chat" in self.dataset_path:
            logger.info("Processing lmsys/toxic-chat dataset")
            
            result = {}
            for split, ds in (dataset.items() if isinstance(dataset, DatasetDict) else [("train", dataset)]):
                # Map from lmsys toxic chat format to our research format
                def map_toxic_chat(example):
                    # Based on SQL output, use user_input instead of conversations
                    query = example.get("user_input", "")
                    
                    # Determine label based on toxicity field (1 = toxic, 0 = not toxic)
                    is_unsafe = example.get("toxicity", 0) > 0
                    
                    # Get the reason from openai_moderation if available
                    reason = example.get("openai_moderation", "") 
                    
                    return {
                        "input": query,
                        "label": "unsafe" if is_unsafe else "safe",
                        "reason": reason
                    }
                
                result[split] = ds.map(map_toxic_chat)
            
            return DatasetDict(result) if isinstance(dataset, DatasetDict) else result["train"]
        else:
            # Default processing
            logger.info("Using default dataset processing")
            
            result = {}
            for split, ds in (dataset.items() if isinstance(dataset, DatasetDict) else [("train", dataset)]):
                # Try to adapt to our research format
                def map_default(example):
                    # Extract input field (adjust based on actual structure)
                    input_text = example.get("prompt", example.get("text", example.get("input", "")))
                    
                    # Try to determine label
                    label = example.get("label", example.get("is_harmful", example.get("is_unsafe", "safe")))
                    if isinstance(label, bool):
                        label = "unsafe" if label else "safe"
                    elif not isinstance(label, str):
                        # Default to safe if we can't determine
                        label = "safe"
                    
                    # Try to extract reason if available
                    reason = example.get("reason", example.get("category", ""))
                    
                    return {
                        "input": input_text,
                        "label": label,
                        "reason": reason
                    }
                
                result[split] = ds.map(map_default)
            
            return DatasetDict(result) if isinstance(dataset, DatasetDict) else result["train"]
    
    def _split_and_save_dataset(self, full_dataset: HFDataset, seed: int = 42) -> None:
        """
        Split the dataset into train, test, and validation sets.
        
        Args:
            full_dataset: Full dataset to split
            seed: Random seed for reproducibility
        """
        # First split: train + rest
        train_ratio = self.split_ratios["train"]
        test_val_ratio = self.split_ratios["test"] + self.split_ratios["validation"]
        
        train_dataset, rest_dataset = train_test_split(
            full_dataset, 
            test_size=test_val_ratio / (train_ratio + test_val_ratio),
            random_state=seed
        )
        
        # Second split: test + validation
        test_ratio = self.split_ratios["test"] / test_val_ratio
        
        test_dataset, validation_dataset = train_test_split(
            rest_dataset,
            test_size=1 - test_ratio,
            random_state=seed
        )
        
        # Convert back to Dataset objects
        train_dataset = HFDataset.from_dict(train_dataset)
        test_dataset = HFDataset.from_dict(test_dataset)
        validation_dataset = HFDataset.from_dict(validation_dataset)
        
        # Create DatasetDict
        self.dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
            "validation": validation_dataset
        })
        
        # Save the splits for future use
        os.makedirs(self.store_dir, exist_ok=True)
        train_dataset.to_json(os.path.join(self.store_dir, "train.jsonl"))
        test_dataset.to_json(os.path.join(self.store_dir, "test.jsonl"))
        validation_dataset.to_json(os.path.join(self.store_dir, "validation.jsonl"))
        
        logger.info(f"Dataset split and saved: train={len(train_dataset)}, test={len(test_dataset)}, validation={len(validation_dataset)}")
    
    def store_blocked_input(self, query: str, input_safe: bool, output_safe: bool, reason: str) -> None:
        """
        Store an input that was blocked by the output filter but not by the input filter.
        
        Args:
            query: User query
            input_safe: Whether the input was deemed safe
            output_safe: Whether the output was deemed safe
            reason: Reason for blocking
        """
        # Only store if input was safe but output was not
        if input_safe and not output_safe:
            item = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query,
                "input_safe": input_safe,
                "output_safe": output_safe,
                "reason": reason
            }
            
            self.blocked_inputs.append(item)
            
            # Append to file
            with open(self.blocked_inputs_path, 'a') as f:
                f.write(json.dumps(item) + "\n")
            
            logger.info(f"Stored blocked input: {reason}")
    
    def get_llama_guard_categories(self) -> List[Category]:
        """
        Get LlamaGuard categories for training.
        
        Returns:
            List of Category objects
        """
        return [
            Category(name="Violence and Hate", 
                     description="Encouraging, glorifying, or promoting violence, harm, or hatred"),
            Category(name="Sexual Content", 
                     description="Explicit sexual content, objectification, or inappropriate content involving minors"),
            Category(name="Criminal Planning", 
                     description="Assistance in planning or executing criminal activities"),
            Category(name="Guns and Illegal Weapons", 
                     description="Detailed instructions on obtaining, creating, or using illegal weapons"),
            Category(name="Regulated Substances", 
                     description="Guidance on procuring or manufacturing illegal drugs or regulated substances"),
            Category(name="Self-Harm", 
                     description="Encouragement or instructions for self-harm or suicide"),
            Category(name="Platform Manipulation", 
                     description="Attempts to manipulate, exploit, or bypass the AI system's safety features"),
            Category(name="Economic Harm", 
                     description="Assistance in causing economic harm, such as fraud or theft"),
            Category(name="Fraud and Deception", 
                     description="Assistance in creating fraudulent materials or executing scams"),
            Category(name="Malware and Security", 
                     description="Assistance in creating malware or exploiting security vulnerabilities")
        ]
    
    def prepare_llama_guard_training_examples(self) -> List[TrainingExample]:
        """
        Prepare training examples for finetuning LlamaGuard using the cookbook format.
        
        Returns:
            List of TrainingExample objects
        """
        training_examples = []
        
        # Map from our collected examples to LlamaGuard's TrainingExample format
        for item in self.blocked_inputs:
            query = item["query"]
            
            # Get category code from reason
            reason = item.get("reason", "")
            category_code = "O1"  # Default to Violence and Hate
            for code in ["O1", "O2", "O3", "O4", "O5", "O6", "I1", "I2", "I3", "I4"]:
                if code in reason:
                    category_code = code
                    break
            
            # Create a TrainingExample
            training_examples.append(
                TrainingExample(
                    prompt=query,
                    response="N/A",  # No response needed for prompt-only examples
                    violated_category_codes=[category_code],
                    label="unsafe",
                    explanation=reason
                )
            )
        
        # Add some safe examples from the dataset
        safe_examples = self.dataset["train"].filter(lambda x: x["label"] == "safe").select(range(min(len(self.blocked_inputs), len(self.dataset["train"]) // 2)))
        
        for example in safe_examples:
            training_examples.append(
                TrainingExample(
                    prompt=example["input"],
                    response="N/A",  # No response needed for prompt-only examples
                    violated_category_codes=[],
                    label="safe",
                    explanation="This query is safe and does not contain harmful content."
                )
            )
        
        return training_examples

class LlamaGuardTrainer:
    """Trainer for fine-tuning Llama Guard on blocked inputs using Meta's cookbook."""
    
    def __init__(self, config: Dict[str, Any], dataset_manager: DatasetManager):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            dataset_manager: Dataset manager
        """
        self.config = config
        self.dataset_manager = dataset_manager
        self.base_model_path = config["llama_guard"]["base_model_path"]
        self.output_dir = config["training"]["output_dir"]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fine_tune(self) -> str:
        """
        Fine-tune Llama Guard on blocked inputs using the official cookbook approach.
        
        Returns:
            Path to the fine-tuned model
        """
        # Get seed from config
        seed = self.config.get("seed", 42)
        
        logger.info(f"Starting fine-tuning of Llama Guard using base model: {self.base_model_path}")
        
        # Create timestamp for this training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"llama_guard_improved_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Prepare training examples using the cookbook format
        training_examples = self.dataset_manager.prepare_llama_guard_training_examples()
        logger.info(f"Prepared {len(training_examples)} training examples")
        
        # Save raw training examples for reference
        with open(os.path.join(output_dir, "training_examples.json"), "w") as f:
            json.dump([{
                "prompt": ex.prompt,
                "response": ex.response,
                "violated_category_codes": ex.violated_category_codes,
                "label": ex.label,
                "explanation": ex.explanation
            } for ex in training_examples], f, indent=2)
        
        # Step 2: Setup LlamaGuard formatting configs from cookbook
        guidelines = Guidelines(
            categories=self.dataset_manager.get_llama_guard_categories(),
            category_code_prefix="O"  # Use "O" prefix as in the original LlamaGuard
        )
        
        llama_guard_prompt_configs = LlamaGuardPromptConfigs(
            instructions_format_string="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.
    <BEGIN UNSAFE CONTENT CATEGORIES>
    {guidelines}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>
    {conversation}
    <END CONVERSATION>

    Provide your safety assessment for $agent_type in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
            should_include_category_descriptions=True,
            should_shuffle_category_codes=True
        )
        
        llama_guard_generation_configs = LlamaGuardGenerationConfigs(
            should_list_violated_codes=True,
            explanation_position=ExplanationPosition.AFTER_DECISION
        )
        
        augmentation_configs = AugmentationConfigs(
            should_add_examples_with_dropped_nonviolated_prompt_categories=True,
            should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
            explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
        )
        
        formatter_configs = FormatterConfigs(
            guidelines=guidelines,
            llama_guard_prompt_configs=llama_guard_prompt_configs,
            llama_guard_generation_configs=llama_guard_generation_configs,
            augmentation_configs=augmentation_configs,
            random_seed=seed
        )
        
        # Step 3: Create formatted examples using the cookbook
        logger.info("Formatting training examples using cookbook")
        formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs)
        
        # Save formatted examples for reference
        with open(os.path.join(output_dir, "formatted_examples.txt"), "w") as f:
            for example in formatted_examples:
                f.write(example + "\n\n---\n\n")
        
        # Step 4: Load base model and tokenizer
        logger.info("Loading base model and tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # Make sure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with 4-bit quantization for training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        
        # NEW CODE: Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # NEW CODE: Define LoRA configuration for PEFT
        lora_config = LoraConfig(
            r=8,                      # Rank dimension
            lora_alpha=32,            # Alpha parameter for LoRA scaling
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,        # Dropout probability for LoRA layers
            bias="none",              # Bias type
            task_type=TaskType.CAUSAL_LM
        )
        
        # NEW CODE: Apply LoRA to model
        logger.info("Applying LoRA adapters to the model")
        model = get_peft_model(model, lora_config)
        
        # Print the model to make sure LoRA modules are attached correctly
        model.print_trainable_parameters()
        
        # Step 5: Tokenize the formatted examples
        logger.info("Tokenizing formatted examples")
        tokenized_examples = []
        for example in formatted_examples:
            # Update the tokenizer call to include padding
            inputs = tokenizer(
                example, 
                return_tensors="pt", 
                padding='max_length',
                truncation=True, 
                max_length=2048
            )
            labels = inputs["input_ids"].clone()

            
            # Set labels for input tokens (non-output) to -100 so they're not included in loss
            # Find the assistant tag location
            assistant_tag = "<|im_start|>assistant"
            example_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            assistant_pos = -1
            
            for i, token in enumerate(example_tokens):
                if assistant_tag in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][i:i+10], skip_special_tokens=False):
                    assistant_pos = i
                    break
            
            if assistant_pos >= 0:
                labels[0, :assistant_pos] = -100
            
            tokenized_examples.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": labels[0]
            })
        
        # Step 6: Create dataset
        train_dataset = HFDataset.from_list(tokenized_examples)
        
        # Step 7: Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            num_train_epochs=self.config["training"]["epochs"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            fp16=True,
            bf16=False,
            weight_decay=self.config["training"]["weight_decay"],
            warmup_steps=self.config["training"]["warmup_steps"],
            lr_scheduler_type="cosine",
            # Set seed for training reproducibility
            seed=seed,
            data_seed=seed,
            # Peft training settings
            remove_unused_columns=False,
            gradient_checkpointing=True
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Step 8: Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Step 9: Train
        logger.info("Starting training")
        trainer.train()
        
        # Step 10: Save model
        logger.info(f"Training complete, saving model to {output_dir}")
        # Save PEFT adapter
        model.save_pretrained(output_dir)
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning complete, model saved to {output_dir}")
        
        return output_dir
        """
        Fine-tune Llama Guard on blocked inputs using the official cookbook approach.
        
        Returns:
            Path to the fine-tuned model
        """
        # Get seed from config
        seed = self.config.get("seed", 42)
        
        logger.info(f"Starting fine-tuning of Llama Guard using base model: {self.base_model_path}")
        
        # Create timestamp for this training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"llama_guard_improved_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Prepare training examples using the cookbook format
        training_examples = self.dataset_manager.prepare_llama_guard_training_examples()
        logger.info(f"Prepared {len(training_examples)} training examples")
        
        # Save raw training examples for reference
        with open(os.path.join(output_dir, "training_examples.json"), "w") as f:
            json.dump([{
                "prompt": ex.prompt,
                "response": ex.response,
                "violated_category_codes": ex.violated_category_codes,
                "label": ex.label,
                "explanation": ex.explanation
            } for ex in training_examples], f, indent=2)
        
        # Step 2: Setup LlamaGuard formatting configs from cookbook
        guidelines = Guidelines(
            categories=self.dataset_manager.get_llama_guard_categories(),
            category_code_prefix="O"  # Use "O" prefix as in the original LlamaGuard
        )
        
        llama_guard_prompt_configs = LlamaGuardPromptConfigs(
            instructions_format_string="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
{guidelines}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
{conversation}
<END CONVERSATION>

Provide your safety assessment for $agent_type in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
            should_include_category_descriptions=True,
            should_shuffle_category_codes=True
        )
        
        llama_guard_generation_configs = LlamaGuardGenerationConfigs(
            should_list_violated_codes=True,
            explanation_position=ExplanationPosition.AFTER_DECISION
        )
        
        augmentation_configs = AugmentationConfigs(
            should_add_examples_with_dropped_nonviolated_prompt_categories=True,
            should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
            explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
        )
        
        formatter_configs = FormatterConfigs(
            guidelines=guidelines,
            llama_guard_prompt_configs=llama_guard_prompt_configs,
            llama_guard_generation_configs=llama_guard_generation_configs,
            augmentation_configs=augmentation_configs,
            random_seed=seed
        )
        
        # Step 3: Create formatted examples using the cookbook
        logger.info("Formatting training examples using cookbook")
        formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs)
        
        # Save formatted examples for reference
        with open(os.path.join(output_dir, "formatted_examples.txt"), "w") as f:
            for example in formatted_examples:
                f.write(example + "\n\n---\n\n")
        
        # Step 4: Load base model and tokenizer
        logger.info("Loading base model and tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # Make sure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with 4-bit quantization for training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        
        # Step 5: Tokenize the formatted examples
        logger.info("Tokenizing formatted examples")
        tokenized_examples = []
        for example in formatted_examples:
            inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=2048)
            labels = inputs["input_ids"].clone()
            
            # Set labels for input tokens (non-output) to -100 so they're not included in loss
            # Find the assistant tag location
            assistant_tag = "<|im_start|>assistant"
            example_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            assistant_pos = -1
            
            for i, token in enumerate(example_tokens):
                if assistant_tag in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][i:i+10], skip_special_tokens=False):
                    assistant_pos = i
                    break
            
            if assistant_pos >= 0:
                labels[0, :assistant_pos] = -100
            
            tokenized_examples.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": labels[0]
            })
        
        # Step 6: Create dataset
        train_dataset = HFDataset.from_list(tokenized_examples)
        
        # Step 7: Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            num_train_epochs=self.config["training"]["epochs"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            fp16=True,
            bf16=False,
            weight_decay=self.config["training"]["weight_decay"],
            warmup_steps=self.config["training"]["warmup_steps"],
            lr_scheduler_type="cosine",
            # Set seed for training reproducibility
            seed=seed,
            data_seed=seed,
            # Peft training settings
            remove_unused_columns=False,
            gradient_checkpointing=True
        )
        
        # Step 8: Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        
        # Step 9: Train
        logger.info("Starting training")
        trainer.train()
        
        # Step 10: Save model
        logger.info(f"Training complete, saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning complete, model saved to {output_dir}")
        
        return output_dir

class Evaluator:
    """Evaluates default and improved Llama Guard models."""
    
    def __init__(self, config: Dict[str, Any], dataset_manager: DatasetManager):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
            dataset_manager: Dataset manager
        """
        self.config = config
        self.dataset_manager = dataset_manager
        self.default_model_path = config["llama_guard"]["default_model_path"]
        self.improved_model_path = config["evaluation"]["improved_model_path"]
        self.results_dir = config["evaluation"]["results_dir"]
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate default and improved Llama Guard models.
        
        Returns:
            Evaluation results
        """
        # Get seed from config
        seed = self.config.get("seed", 42)
        
        logger.info("Starting evaluation of Llama Guard models")
        
        # Load default model
        default_guard = LlamaGuardFilter(self.default_model_path)
        
        # Check if improved model path exists
        if not os.path.exists(self.improved_model_path):
            logger.error(f"Improved model path does not exist: {self.improved_model_path}")
            raise ValueError(f"Improved model path does not exist: {self.improved_model_path}")
        
        # Load improved model
        logger.info(f"Loading improved model from {self.improved_model_path}")
        improved_guard = LlamaGuardFilter(self.improved_model_path)
        
        # Get test dataset
        test_dataset = self.dataset_manager.dataset["test"]
        
        # Use a consistent seed for sample selection
        random.seed(seed)
        
        # Select a subset for evaluation if test set is large
        eval_size = min(len(test_dataset), self.config["evaluation"]["max_eval_samples"])
        eval_indices = random.sample(range(len(test_dataset)), eval_size)
        eval_dataset = test_dataset.select(eval_indices)
        
        logger.info(f"Evaluating on {eval_size} samples from test set")
        
        # Prepare results
        results = {
            "default_model": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "predictions": []
            },
            "improved_model": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "predictions": []
            }
        }
        
        # Evaluate
        for i, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            input_text = example["input"]
            true_label = example["label"] == "unsafe"  # True if unsafe
            
            # Evaluate default model
            default_safe, default_reason = default_guard.check_content(input_text)
            default_prediction = not default_safe  # Model predicts unsafe
            
            # Update default model stats
            if default_prediction and true_label:
                results["default_model"]["true_positives"] += 1
            elif default_prediction and not true_label:
                results["default_model"]["false_positives"] += 1
            elif not default_prediction and not true_label:
                results["default_model"]["true_negatives"] += 1
            elif not default_prediction and true_label:
                results["default_model"]["false_negatives"] += 1
            
            results["default_model"]["predictions"].append({
                "input": input_text,
                "true_label": "unsafe" if true_label else "safe",
                "predicted_label": "unsafe" if default_prediction else "safe",
                "reason": default_reason
            })
            
            # Evaluate improved model
            improved_safe, improved_reason = improved_guard.check_content(input_text)
            improved_prediction = not improved_safe  # Model predicts unsafe
            
            # Update improved model stats
            if improved_prediction and true_label:
                results["improved_model"]["true_positives"] += 1
            elif improved_prediction and not true_label:
                results["improved_model"]["false_positives"] += 1
            elif not improved_prediction and not true_label:
                results["improved_model"]["true_negatives"] += 1
            elif not improved_prediction and true_label:
                results["improved_model"]["false_negatives"] += 1
            
            results["improved_model"]["predictions"].append({
                "input": input_text,
                "true_label": "unsafe" if true_label else "safe",
                "predicted_label": "unsafe" if improved_prediction else "safe",
                "reason": improved_reason
            })
        
        # Calculate metrics
        for model_name in ["default_model", "improved_model"]:
            tp = results[model_name]["true_positives"]
            fp = results[model_name]["false_positives"]
            tn = results[model_name]["true_negatives"]
            fn = results[model_name]["false_negatives"]
            
            # Accuracy
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[model_name]["metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete, results saved to {results_path}")
        
        # Return summarized results
        summary = {
            "default_model": results["default_model"]["metrics"],
            "improved_model": results["improved_model"]["metrics"],
            "improvement": {
                "accuracy": results["improved_model"]["metrics"]["accuracy"] - results["default_model"]["metrics"]["accuracy"],
                "precision": results["improved_model"]["metrics"]["precision"] - results["default_model"]["metrics"]["precision"],
                "recall": results["improved_model"]["metrics"]["recall"] - results["default_model"]["metrics"]["recall"],
                "f1": results["improved_model"]["metrics"]["f1"] - results["default_model"]["metrics"]["f1"]
            }
        }
        
        return summary

class ResearchSystem:
    """Main class for the research experiment."""
    
    def __init__(self, config_path: str):
        """
        Initialize the research system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set global seed for reproducibility
        seed = self.config.get("seed", 42)
        self._set_seed(seed)
        logger.info(f"Set global seed to {seed} for reproducibility")
        
        # Initialize components
        self.dataset_manager = DatasetManager(self.config)
        
        # Initialize filters and model if in process mode
        if self.config["mode"] == "process":
            self.input_filter = LlamaGuardFilter(self.config["llama_guard"]["default_model_path"])
            self.llm = Llama8BModel(self.config["llama"]["model_path"])
            self.output_filter = LlamaGuardFilter(self.config["llama_guard"]["default_model_path"])
        
        logger.info("Research system initialized")
    
    def _set_seed(self, seed: int) -> None:
        """
        Set seed for all random operations to ensure reproducibility.
        
        Args:
            seed: Seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Set Python hash seed for reproducible dict iterations
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def process_query(self, query: str) -> Tuple[str, bool, bool, str]:
        """
        Process a query through the system pipeline.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (response, input_safe, output_safe, reason)
        """
        # Check input with Llama Guard
        input_safe, input_reason = self.input_filter.check_content(query)
        
        # If input is not safe, return early
        if not input_safe:
            return "", False, False, input_reason
        
        # Generate response with model
        response = self.llm.generate_response(query)
        
        # Check output with Llama Guard
        output_safe, output_reason = self.output_filter.check_content(response)
        
        # If output is not safe, store the query for later training
        if not output_safe:
            self.dataset_manager.store_blocked_input(query, input_safe, output_safe, output_reason)
        
        return response, input_safe, output_safe, output_reason if not output_safe else input_reason
    
    def process_dataset(self, split: str = "train", sample_size: int = None) -> Dict[str, Any]:
        """
        Process queries from the dataset through the system pipeline.
        
        Args:
            split: Dataset split to use
            sample_size: Number of samples to process (None for all)
            
        Returns:
            Processing statistics
        """
        dataset = self.dataset_manager.dataset[split]
        
        if sample_size is not None:
            # Select a random subset
            indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
            dataset = dataset.select(indices)
        
        stats = {
            "total": len(dataset),
            "input_blocked": 0,
            "output_blocked": 0,
            "both_safe": 0
        }
        
        logger.info(f"Processing {len(dataset)} samples from {split} split")
        
        for i, example in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            input_text = example["input"]
            
            # Process through the pipeline
            response, input_safe, output_safe, reason = self.process_query(input_text)
            
            # Update stats
            if not input_safe:
                stats["input_blocked"] += 1
            elif not output_safe:
                stats["output_blocked"] += 1
            else:
                stats["both_safe"] += 1
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")
        
        # Calculate percentages
        stats["input_blocked_percent"] = (stats["input_blocked"] / stats["total"]) * 100
        stats["output_blocked_percent"] = (stats["output_blocked"] / stats["total"]) * 100
        stats["both_safe_percent"] = (stats["both_safe"] / stats["total"]) * 100
        
        logger.info(f"Processing complete: {stats}")
        
        return stats
    
    def train_improved_model(self) -> str:
        """
        Train an improved Llama Guard model using collected blocked inputs.
        
        Returns:
            Path to the improved model
        """
        trainer = LlamaGuardTrainer(self.config, self.dataset_manager)
        model_path = trainer.fine_tune()
        
        # Update config with improved model path
        self.config["evaluation"]["improved_model_path"] = model_path
        
        return model_path
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate default and improved Llama Guard models.
        
        Returns:
            Evaluation results
        """
        evaluator = Evaluator(self.config, self.dataset_manager)
        results = evaluator.evaluate()
        
        return results
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run the full research experiment.
        
        Returns:
            Experiment results
        """
        results = {}
        
        # Step 1: Process dataset to collect blocked inputs
        logger.info("Step 1: Processing dataset to collect blocked inputs")
        process_stats = self.process_dataset(
            split=self.config["experiment"]["process_split"],
            sample_size=self.config["experiment"]["process_sample_size"]
        )
        results["process_stats"] = process_stats
        
        # Step 2: Train improved model
        logger.info("Step 2: Training improved Llama Guard model")
        improved_model_path = self.train_improved_model()
        results["improved_model_path"] = improved_model_path
        
        # Step 3: Evaluate models
        logger.info("Step 3: Evaluating default and improved models")
        eval_results = self.evaluate_models()
        results["evaluation"] = eval_results
        
        # Save full results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config["experiment"]["results_dir"], f"experiment_results_{timestamp}.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Full experiment complete, results saved to {results_path}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama Guard Improvement Research")
    parser.add_argument("--config", type=str, default="research_config.json", help="Path to configuration file")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--process-dataset", action="store_true", help="Process dataset to collect blocked inputs")
    parser.add_argument("--train", action="store_true", help="Train improved Llama Guard model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate default and improved models")
    parser.add_argument("--full-experiment", action="store_true", help="Run full research experiment")
    
    args = parser.parse_args()
    
    system = ResearchSystem(args.config)
    
    if args.query:
        response, input_safe, output_safe, reason = system.process_query(args.query)
        print(f"Input Safe: {input_safe}")
        print(f"Output Safe: {output_safe}")
        print(f"Reason: {reason}")
        if input_safe:
            print(f"Response: {response}")
    
    if args.process_dataset:
        stats = system.process_dataset(
            split=system.config["experiment"]["process_split"],
            sample_size=system.config["experiment"]["process_sample_size"]
        )
        print(json.dumps(stats, indent=2))
    
    if args.train:
        model_path = system.train_improved_model()
        print(f"Improved model saved to: {model_path}")
    
    if args.evaluate:
        results = system.evaluate_models()
        print(json.dumps(results, indent=2))
    
    if args.full_experiment:
        results = system.run_full_experiment()
        print(json.dumps(results, indent=2))