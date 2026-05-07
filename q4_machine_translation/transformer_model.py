"""
Transformer-based model for machine translation (Q4).
Uses T5-small.
"""
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
)
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class TransformerMT:
    """
    Transformer-based machine translation using T5.
    T5 uses HuggingFace tokenizers .
    """
    
    def __init__(self, model_name: str = "t5-small",
                 max_length: int = 128):
        """
        Initialize transformer MT model.
        
        Args:
            model_name: Pre-trained model name (t5-small, t5-base)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"Loading translation model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    
    def translate(self, text: str, source_lang: str = "English", 
                  target_lang: str = "German") -> str:
        """
        Translate a single text.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translated text
        """
        # T5 prompt format
        input_text = f"translate {source_lang} to {target_lang}: {text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        translation = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return translation
    
    def translate_batch(self, texts: List[str], 
                        source_lang: str = "English",
                        target_lang: str = "German",
                        batch_size: int = 16) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of source texts
            source_lang: Source language
            target_lang: Target language
            batch_size: Batch size
            
        Returns:
            List of translations
        """
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # T5 prompt format for each text
            input_texts = [f"translate {source_lang} to {target_lang}: {t}" for t in batch]
            
            inputs = self.tokenizer(
                input_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            batch_translations = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            translations.extend(batch_translations)
        
        return translations