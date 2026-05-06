"""
Abstractive summarization module (Q3).
Implements BART/T5-based summarization.
"""
import torch
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class BARTSummarizer:
    """
    BART-based abstractive summarization.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn",
                 max_length: int = 130, min_length: int = 30):
        """
        Initialize BART summarizer.
        
        Args:
            model_name: Pre-trained BART model name
            max_length: Maximum summary length
            min_length: Minimum summary length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        print(f"Loading BART model: {model_name}...")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name
        ).to(DEVICE)
        
        # Also initialize using pipeline for simpler usage
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def summarize(self, text: str, max_input_length: int = 1024) -> str:
        """
        Generate abstractive summary.
        
        Args:
            text: Input text
            max_input_length: Maximum input length
            
        Returns:
            Generated summary
        """
        # Truncate input if necessary
        inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        summary = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )
        
        return summary
    
    def summarize_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text)
            summaries.append(summary)
        return summaries


