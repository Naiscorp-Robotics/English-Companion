import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, List, Tuple, Optional, Union, Any

class BertErrorClassifier(nn.Module):
    """
    BERT-based model for classifying grammatical and vocabulary errors in English text
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,  # correct, vocabulary error, grammar error
        dropout_rate: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Initialize the error classifier model
        
        Args:
            model_name: The BERT model name to use
            num_classes: Number of error classes to predict
            dropout_rate: Dropout probability
            freeze_bert: Whether to freeze BERT parameters during training
        """
        super(BertErrorClassifier, self).__init__()
        
        # Load BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Classification logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
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
                'num_classes': self.classifier.out_features,
                'hidden_size': self.hidden_size
            }
        }, path)
        
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BertErrorClassifier":
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
            num_classes=config.get('num_classes', 3),
            **kwargs
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 