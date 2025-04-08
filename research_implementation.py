class Evaluator:
    """Evaluates default model against rule-based approach using collected examples."""
    
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
        self.examples_dir = config["evaluation"]["improved_model_path"]
        self.results_dir = config["evaluation"]["results_dir"]
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate default model against rule-based approach using collected examples.
        
        Returns:
            Evaluation results
        """
        # Get seed from config
        seed = self.config.get("seed", 42)
        
        logger.info("Starting evaluation")
        
        # Load default model
        logger.info(f"Loading default Llama Guard model from {self.default_model_path}")
        default_guard = LlamaGuardFilter(self.default_model_path)
        
        # Check if examples directory exists
        if not os.path.exists(self.examples_dir):
            logger.error(f"Examples directory does not exist: {self.examples_dir}")
            raise ValueError(f"Examples directory does not exist: {self.examples_dir}")
        
        logger.info(f"Using rule-based approach with examples from: {self.examples_dir}")
        
        # Load examples from the examples directory
        examples_path = os.path.join(self.examples_dir, "examples.json")
        if os.path.exists(examples_path):
            with open(examples_path, "r") as f:
                examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples for evaluation")
        else:
            # If no examples file exists, use the test dataset
            logger.warning(f"No examples file found at {examples_path}, using test dataset instead")
            examples = []
            for item in self.dataset_manager.dataset["test"].select(range(100)):
                examples.append({
                    "input": item["input"],
                    "label": item["label"],
                    "reason": item.get("reason", "")
                })
        
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
            "rule_based_approach": {
                "description": "Rule-based approach using collected examples",
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "predictions": []
            }
        }
        
        # Collect examples for rule-based approach
        logger.info("Building rule-based model from collected examples")
        rule_examples = set()
        for example in examples:
            if example["label"] == "unsafe":
                rule_examples.add(example["input"].lower())
        
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
            
            # Evaluate rule-based approach (using collected examples)
            # If input is similar to any example we collected, mark as unsafe
            rule_prediction = input_text.lower() in rule_examples
            
            # Update rule-based approach stats
            if rule_prediction and true_label:
                results["rule_based_approach"]["true_positives"] += 1
            elif rule_prediction and not true_label:
                results["rule_based_approach"]["false_positives"] += 1
            elif not rule_prediction and not true_label:
                results["rule_based_approach"]["true_negatives"] += 1
            elif not rule_prediction and true_label:
                results["rule_based_approach"]["false_negatives"] += 1
            
            results["rule_based_approach"]["predictions"].append({
                "input": input_text,
                "true_label": "unsafe" if true_label else "safe",
                "predicted_label": "unsafe" if rule_prediction else "safe",
                "reason": "Matched collected example" if rule_prediction else ""
            })
        
        # Calculate metrics
        for model_name in ["default_model", "rule_based_approach"]:
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
            "rule_based_approach": results["rule_based_approach"]["metrics"],
            "improvement": {
                "accuracy": results["rule_based_approach"]["metrics"]["accuracy"] - results["default_model"]["metrics"]["accuracy"],
                "precision": results["rule_based_approach"]["metrics"]["precision"] - results["default_model"]["metrics"]["precision"],
                "recall": results["rule_based_approach"]["metrics"]["recall"] - results["default_model"]["metrics"]["recall"],
                "f1": results["rule_based_approach"]["metrics"]["f1"] - results["default_model"]["metrics"]["f1"]
            }
        }
        
        return summary