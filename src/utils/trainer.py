import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import os
import time
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Trainer:
    """
    Trainer class for training and evaluating models
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs",
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.01
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to use for training
            output_dir: Directory to save outputs
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            save_steps: Steps between saving model
            early_stopping_patience: Patience for early stopping
            early_stopping_threshold: Threshold for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # Default optimizer if not provided
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            model.parameters(), lr=2e-5, weight_decay=0.01
        )
        
        # Default criterion if not provided
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
    
    def train(self, epochs: int = 3):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting training on device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training loop
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                if "labels" in batch:
                    # Classification task
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids")
                    )
                    loss = self.criterion(outputs, batch["labels"])
                elif "token_labels" in batch:
                    # Token classification task
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids")
                    )
                    
                    # Reshape outputs for token classification
                    active_loss = batch["token_labels"] != -100
                    active_logits = outputs.view(-1, outputs.shape[-1])
                    active_labels = torch.where(
                        active_loss,
                        batch["token_labels"].view(-1),
                        torch.tensor(self.criterion.ignore_index).type_as(batch["token_labels"])
                    )
                    loss = self.criterion(active_logits, active_labels)
                else:
                    raise ValueError("Batch must contain either 'labels' or 'token_labels'")
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": self.global_step
                })
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self.training_history["train_loss"].append({
                        "step": self.global_step,
                        "loss": loss.item()
                    })
                
                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    val_loss, val_metrics = self.evaluate(self.val_loader)
                    
                    self.training_history["val_loss"].append({
                        "step": self.global_step,
                        "loss": val_loss
                    })
                    
                    self.training_history["val_metrics"].append({
                        "step": self.global_step,
                        "metrics": val_metrics
                    })
                    
                    # Early stopping check
                    if val_loss < self.best_val_loss - self.early_stopping_threshold:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        
                        # Save best model
                        self.save_model(os.path.join(self.output_dir, "best_model.pt"))
                    else:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping at step {self.global_step}")
                            break
                
                # Save model
                if self.global_step % self.save_steps == 0:
                    self.save_model(os.path.join(self.output_dir, f"model_step_{self.global_step}.pt"))
            
            # End of epoch
            avg_train_loss = train_loss / len(self.train_loader)
            print(f"Average train loss: {avg_train_loss:.4f}")
            
            # Check for early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered")
                break
        
        # End of training
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            print("Evaluating on test set...")
            test_loss, test_metrics = self.evaluate(self.test_loader)
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test metrics: {test_metrics}")
            
            # Save test results
            with open(os.path.join(self.output_dir, "test_results.json"), "w") as f:
                json.dump({
                    "loss": test_loss,
                    "metrics": test_metrics
                }, f, indent=2)
        
        # Save training history
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model.pt"))
        
        return self.training_history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.eval()
        eval_loss = 0
        all_preds = []
        all_labels = []
        all_token_preds = []
        all_token_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                if "labels" in batch:
                    # Classification task
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids")
                    )
                    loss = self.criterion(outputs, batch["labels"])
                    
                    # Get predictions
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
                    
                elif "token_labels" in batch:
                    # Token classification task
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids")
                    )
                    
                    # Reshape outputs for token classification
                    active_loss = batch["token_labels"] != -100
                    active_logits = outputs.view(-1, outputs.shape[-1])
                    active_labels = torch.where(
                        active_loss,
                        batch["token_labels"].view(-1),
                        torch.tensor(self.criterion.ignore_index).type_as(batch["token_labels"])
                    )
                    loss = self.criterion(active_logits, active_labels)
                    
                    # Get predictions
                    preds = torch.argmax(outputs, dim=-1)
                    
                    # Only consider tokens that are not padding
                    for i in range(batch["token_labels"].shape[0]):
                        for j in range(batch["token_labels"].shape[1]):
                            if batch["token_labels"][i, j] != -100:
                                all_token_preds.append(preds[i, j].item())
                                all_token_labels.append(batch["token_labels"][i, j].item())
                else:
                    raise ValueError("Batch must contain either 'labels' or 'token_labels'")
                
                eval_loss += loss.item()
        
        # Calculate average loss
        avg_loss = eval_loss / len(data_loader)
        
        # Calculate metrics
        metrics = {}
        if all_preds and all_labels:
            # Classification metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
            conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix
            }
        elif all_token_preds and all_token_labels:
            # Token classification metrics
            accuracy = accuracy_score(all_token_labels, all_token_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_token_labels, all_token_preds, average='weighted'
            )
            conf_matrix = confusion_matrix(all_token_labels, all_token_preds).tolist()
            
            metrics = {
                "token_accuracy": accuracy,
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": f1,
                "token_confusion_matrix": conf_matrix
            }
        
        self.model.train()
        return avg_loss, metrics
    
    def save_model(self, path):
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        print(f"Model loaded from {path}")
        
        return self.model 