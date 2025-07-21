import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, List, Tuple, Optional, Union, Any

class BertErrorDetector(nn.Module):
    """
    BERT-based model for detecting error words in English text
    This is a token classification task (similar to NER)
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,  # 0: correct, 1: error
        dropout_rate: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Initialize the error detector model
        
        Args:
            model_name: The BERT model name to use
            num_labels: Number of labels to predict (typically binary: error or not)
            dropout_rate: Dropout probability
            freeze_bert: Whether to freeze BERT parameters during training
        """
        super(BertErrorDetector, self).__init__()
        
        # Load BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Token classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Token classification logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the last hidden state for token classification
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classify each token
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits
    
    def predict_error_tokens(self, logits, input_ids, tokenizer, threshold=0.5):
        """
        Predict which tokens are errors
        
        Args:
            logits: Model output logits
            input_ids: Input token IDs
            tokenizer: Tokenizer used for encoding
            threshold: Probability threshold for error detection
            
        Returns:
            List of error words and their positions
        """
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get error probabilities (class 1)
        error_probs = probs[:, :, 1]
        
        # Create mask for tokens above threshold
        error_mask = error_probs > threshold
        
        # Get original tokens
        batch_size = input_ids.shape[0]
        error_words = []
        
        for b in range(batch_size):
            sample_errors = []
            mask = error_mask[b]
            ids = input_ids[b]
            
            # Get word IDs to group subwords
            word_ids = tokenizer.get_word_ids({"input_ids": ids.unsqueeze(0)})
            
            # Track which words have been marked as errors
            error_word_ids = set()
            
            for i, (is_error, token_id, word_id) in enumerate(zip(mask, ids, word_ids)):
                if is_error and word_id is not None and word_id not in error_word_ids:
                    # Find all tokens for this word
                    word_tokens = []
                    for j, wid in enumerate(word_ids):
                        if wid == word_id:
                            word_tokens.append(j)
                    
                    # Get the full word
                    word = tokenizer.decode(ids[word_tokens])
                    sample_errors.append((word, word_id))
                    error_word_ids.add(word_id)
            
            error_words.append(sample_errors)
            
        return error_words
    
    def save_pretrained(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': self.bert.config._name_or_path,
                'num_labels': self.classifier.out_features,
                'hidden_size': self.hidden_size
            }
        }, path)
        
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BertErrorDetector":
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
            **kwargs: Additional arguments
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        
        model = cls(
            model_name=config.get('model_name', 'bert-base-uncased'),
            num_labels=config.get('num_labels', 2),
            **kwargs
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 