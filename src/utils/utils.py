import torch
import numpy as np
import random
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    """
    Get device for PyTorch
    
    Returns:
        Device string
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_json(data: Any, path: str) -> None:
    """
    Save data as JSON
    
    Args:
        data: Data to save
        path: Path to save the data
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path: str) -> Any:
    """
    Load data from JSON
    
    Args:
        path: Path to load the data from
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    train_steps = [item['step'] for item in history['train_loss']]
    train_losses = [item['loss'] for item in history['train_loss']]
    plt.plot(train_steps, train_losses, label='Train')
    
    # Plot validation loss
    val_steps = [item['step'] for item in history['val_loss']]
    val_losses = [item['loss'] for item in history['val_loss']]
    plt.plot(val_steps, val_losses, label='Validation')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation metrics
    plt.subplot(1, 2, 2)
    val_steps = [item['step'] for item in history['val_metrics']]
    
    # Check which metrics are available
    if 'accuracy' in history['val_metrics'][0]['metrics']:
        # Classification metrics
        val_acc = [item['metrics']['accuracy'] for item in history['val_metrics']]
        val_f1 = [item['metrics']['f1'] for item in history['val_metrics']]
        
        plt.plot(val_steps, val_acc, label='Accuracy')
        plt.plot(val_steps, val_f1, label='F1 Score')
    elif 'token_accuracy' in history['val_metrics'][0]['metrics']:
        # Token classification metrics
        val_acc = [item['metrics']['token_accuracy'] for item in history['val_metrics']]
        val_f1 = [item['metrics']['token_f1'] for item in history['val_metrics']]
        
        plt.plot(val_steps, val_acc, label='Token Accuracy')
        plt.plot(val_steps, val_f1, label='Token F1 Score')
    
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_classification_report(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict:
    """
    Generate classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as dictionary
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_error_examples(texts: List[str], true_labels: List[int], pred_labels: List[int], 
                      class_names: List[str], n_examples: int = 10) -> List[Dict]:
    """
    Get examples of errors made by the model
    
    Args:
        texts: List of input texts
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        n_examples: Number of examples to return
        
    Returns:
        List of error examples
    """
    errors = []
    
    for text, true, pred in zip(texts, true_labels, pred_labels):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": class_names[true],
                "pred_label": class_names[pred]
            })
            
            if len(errors) >= n_examples:
                break
                
    return errors 