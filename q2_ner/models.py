"""
Model implementations for Named Entity Recognition (Q2).
Includes CRF, BiLSTM-CRF, and BERT-based models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, SEED

torch.manual_seed(SEED)



class NERBERTModel:
    """
    BERT-based NER model using HuggingFace Transformers.
    Supports layer freezing for efficient fine-tuning.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 num_labels: int = 9,
                 freeze_bert: bool = True,
                 freeze_embeddings: bool = True,
                 unfreeze_last_n: int = 2):
        """
        Initialize BERT NER model.
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of NER tags (including BIO variants)
            freeze_bert: Whether to freeze BERT layers
            freeze_embeddings: Whether to freeze embedding layer
            unfreeze_last_n: Number of last encoder layers to unfreeze
        """
        self.model_name = model_name
        self.device = DEVICE
        
        print(f"Loading {model_name} for token classification...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Apply freezing strategy
        if freeze_bert:
            self._freeze_layers(freeze_embeddings, unfreeze_last_n)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Print trainable params
        self._print_trainable_params()
    
    def _freeze_layers(self, freeze_embeddings: bool = True, unfreeze_last_n: int = 2):
        """
        Strategic layer freezing for BERT.
        
        Freezes embeddings + first layers, keeps last N layers + classifier trainable.
        """
        # 1. Freeze embeddings
        if freeze_embeddings:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            print("✓ Frozen: Embeddings layer")
        
        # 2. Freeze encoder layers selectively
        total_layers = len(self.model.bert.encoder.layer)
        trainable_start = total_layers - unfreeze_last_n
        
        for i, layer in enumerate(self.model.bert.encoder.layer):
            freeze = i < trainable_start
            for param in layer.parameters():
                param.requires_grad = not freeze
            
            if freeze:
                print(f"  Frozen: Encoder layer {i}")
            else:
                print(f"  Trainable: Encoder layer {i}")
        
        # 3. Freeze pooler
        if hasattr(self.model.bert, 'pooler') and self.model.bert.pooler is not None:
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = False
            print("✓ Frozen: Pooler layer")
        
        # 4. Classifier remains trainable
        print("✓ Trainable: Token classification head")
    
    def _print_trainable_params(self):
        """Print parameter summary."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n{'='*40}")
        print(f"BERT NER Parameter Summary:")
        print(f"  Total params:    {total:,}")
        print(f"  Trainable:       {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Frozen:          {total-trainable:,} ({100-trainable/total*100:.1f}%)")
        print(f"{'='*40}\n")
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader, id_to_tag: Dict[int, str]) -> Tuple[List, List]:
        """
        Evaluate the model.
        
        Args:
            data_loader: DataLoader for evaluation data
            id_to_tag: Mapping from ID to tag name
            
        Returns:
            Tuple of (predicted_labels, true_labels)
        """
        self.model.eval()
        true_labels_list = []
        predictions_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)
                
                # Filter -100 and padding
                for pred_seq, true_seq, mask in zip(preds, labels, attention_mask):
                    active_mask = mask.bool() & (true_seq != -100)
                    
                    pred_active = pred_seq[active_mask].cpu().numpy()
                    true_active = true_seq[active_mask].cpu().numpy()
                    
                    pred_tags = [id_to_tag.get(p, 'O') for p in pred_active]
                    true_tags = [id_to_tag.get(t, 'O') for t in true_active]
                    
                    predictions_list.append(pred_tags)
                    true_labels_list.append(true_tags)
        
        return predictions_list, true_labels_list