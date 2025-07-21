import torch
from transformers import BertTokenizer, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any

class BertEnglishTokenizer:
    """
    Wrapper for BERT tokenizer specifically for English text processing
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize the BERT tokenizer
        
        Args:
            model_name: The BERT model name to use
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences longer than max_length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids
        """
        encoded = self.tokenizer(
            text,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return encoded
        
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size
        
        Returns:
            Size of vocabulary
        """
        return len(self.tokenizer)
    
    def get_word_ids(self, encoded_inputs: Dict[str, torch.Tensor]) -> List[List[Optional[int]]]:
        """
        Get word IDs for token-to-word mapping
        
        Args:
            encoded_inputs: The encoded inputs from tokenizer
            
        Returns:
            List of word IDs for each token
        """
        return self.tokenizer.word_ids(batch_index=0)
    
    def save_pretrained(self, path: str) -> None:
        """
        Save the tokenizer to disk
        
        Args:
            path: Path to save the tokenizer
        """
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BertEnglishTokenizer":
        """
        Load tokenizer from disk
        
        Args:
            path: Path to load the tokenizer from
            **kwargs: Additional arguments
            
        Returns:
            Loaded tokenizer
        """
        instance = cls(**kwargs)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        return instance 