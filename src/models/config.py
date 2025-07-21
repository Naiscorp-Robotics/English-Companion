from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_steps: int = 0
    seed: int = 42
    dropout_rate: float = 0.1
    
@dataclass
class ErrorClassifierConfig(ModelConfig):
    """Configuration for error classification model"""
    num_classes: int = 3  # correct, vocabulary error, grammar error
    freeze_bert: bool = False
    class_weights: Optional[List[float]] = None
    
@dataclass
class ErrorDetectorConfig(ModelConfig):
    """Configuration for error detection model"""
    num_labels: int = 2  # correct, error
    freeze_bert: bool = False
    threshold: float = 0.5
    
@dataclass
class TrainingConfig:
    """Configuration for training process"""
    output_dir: str = "outputs"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config_to_json(config: Any, config_path: str) -> None:
    """Save configuration to JSON file"""
    import json
    import dataclasses
    
    # Convert dataclass to dict
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        config_dict = dataclasses.asdict(config)
    else:
        config_dict = config
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2) 