import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import json
import os

class ErrorClassifierPredictor:
    """
    Predictor for error classification
    """
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        label_map: Optional[Dict[int, str]] = None
    ):
        """
        Initialize predictor
        
        Args:
            model: Model for prediction
            tokenizer: Tokenizer for encoding texts
            device: Device to use for prediction
            label_map: Mapping from integer labels to string labels
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map or {
            0: "correct",
            1: "vocab_error",
            2: "gram_error"
        }
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = False,
        batch_size: int = 8
    ) -> Union[List[str], Tuple[List[str], List[Dict[str, float]]]]:
        """
        Make predictions for input texts
        
        Args:
            texts: Input text or list of texts
            return_probabilities: Whether to return probabilities
            batch_size: Batch size for prediction
            
        Returns:
            List of predicted labels or tuple of (labels, probabilities)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted labels
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend([self.label_map[pred] for pred in predictions])
            
            # Get probabilities if requested
            if return_probabilities:
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                for prob in probs:
                    all_probabilities.append({
                        self.label_map[i]: float(p) for i, p in enumerate(prob)
                    })
        
        if return_probabilities:
            return all_predictions, all_probabilities
        else:
            return all_predictions
    
    def predict_batch(self, batch):
        """
        Make predictions for a batch of inputs
        
        Args:
            batch: Batch of inputs
            
        Returns:
            Predictions
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids")
            )
        
        # Get predicted labels
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        label_map_path: Optional[str] = None
    ):
        """
        Load predictor from pretrained model
        
        Args:
            model_path: Path to model
            tokenizer: Tokenizer for encoding texts
            device: Device to use for prediction
            label_map_path: Path to label map JSON file
            
        Returns:
            Loaded predictor
        """
        # Load model
        model = torch.load(model_path, map_location=device)
        
        # Load label map if provided
        label_map = None
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
                
                # Convert string keys to integers
                label_map = {int(k): v for k, v in label_map.items()}
        
        return cls(model, tokenizer, device, label_map)


class ErrorDetectorPredictor:
    """
    Predictor for error word detection
    """
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize predictor
        
        Args:
            model: Model for prediction
            tokenizer: Tokenizer for encoding texts
            device: Device to use for prediction
            threshold: Probability threshold for error detection
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8
    ) -> List[List[Dict[str, Any]]]:
        """
        Make predictions for input texts
        
        Args:
            texts: Input text or list of texts
            batch_size: Batch size for prediction
            
        Returns:
            List of error words for each text
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        all_error_words = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get error words
            batch_error_words = self._extract_error_words(outputs, inputs["input_ids"], batch_texts)
            all_error_words.extend(batch_error_words)
        
        return all_error_words
    
    def _extract_error_words(self, logits, input_ids, original_texts):
        """
        Extract error words from model outputs
        
        Args:
            logits: Model output logits
            input_ids: Input token IDs
            original_texts: Original input texts
            
        Returns:
            List of error words for each text
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get error probabilities (class 1)
        error_probs = probs[:, :, 1]
        
        # Create mask for tokens above threshold
        error_mask = error_probs > self.threshold
        
        batch_size = input_ids.shape[0]
        all_error_words = []
        
        for b in range(batch_size):
            text = original_texts[b]
            error_words = []
            
            # Get tokens from the text
            tokens = self.tokenizer.tokenize(text)
            
            # Get error tokens
            for i, is_error in enumerate(error_mask[b]):
                if i >= len(tokens):
                    break
                    
                if is_error:
                    token = tokens[i]
                    # Skip special tokens
                    if token.startswith('[') and token.endswith(']'):
                        continue
                        
                    # Get the original word
                    word = self.tokenizer.convert_tokens_to_string([token])
                    
                    error_words.append({
                        "word": word,
                        "confidence": float(error_probs[b, i])
                    })
            
            all_error_words.append(error_words)
        
        return all_error_words
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        Load predictor from pretrained model
        
        Args:
            model_path: Path to model
            tokenizer: Tokenizer for encoding texts
            device: Device to use for prediction
            threshold: Probability threshold for error detection
            
        Returns:
            Loaded predictor
        """
        # Load model
        model = torch.load(model_path, map_location=device)
        
        return cls(model, tokenizer, device, threshold) 