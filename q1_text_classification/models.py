"""
Model implementations for text classification (Q1).
Includes: Logistic Regression (TF-IDF), BiLSTM, and BERT-based models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, SEED

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class TFIDFClassifier:
    """
    Classical text classifier using TF-IDF features.
    Supports both Logistic Regression and SVM.
    """
    
    def __init__(self, model_type: str = "logistic_regression", 
                 max_features: int = 5000, C: float = 1.0):
        """
        Initialize the classifier.
        
        Args:
            model_type: "logistic_regression" or "svm"
            max_features: Number of TF-IDF features
            C: Regularization parameter
        """
        self.model_type = model_type
        self.max_features = max_features
        self.C = C
        
        if model_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=C, 
                max_iter=1000, 
                random_state=SEED,
                n_jobs=-1
            )
        elif model_type == "svm":
            self.classifier = SVC(C=C, kernel='linear', random_state=SEED)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier."""
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.classifier.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        return self.classifier.predict_proba(X_test)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100,  
                 hidden_dim: int = 128,  
                 num_layers: int = 2, 
                 num_classes: int = 2, 
                 dropout: float = 0.5,
                 pad_idx: int = 0):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output: hidden_dim * 2 (bidirectional) → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        # hidden: [num_layers*2, batch, hidden_dim]
        last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=-1)
        # last_hidden: [batch, hidden_dim*2]
        
        out = self.classifier(last_hidden)
        return out

class BERTClassifier:
    """
    BERT-based classifier using HuggingFace Transformers.
    Supports feature extraction (frozen BERT) and full fine-tuning.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 num_classes: int = 2, 
                 freeze_bert: bool = True,        # Default: freeze BERT layers
                 freeze_embeddings: bool = True,   # Freeze embedding layer
                 unfreeze_last_n: int = 2):        # Unfreeze last N encoder layers
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of output classes
            freeze_bert: Whether to freeze BERT parameters
            freeze_embeddings: Whether to freeze embedding layer
            unfreeze_last_n: Number of last encoder layers to unfreeze (if freeze_bert=True)
                           0 = freeze all, 2 = unfreeze last 2 layers (recommended)
        """
        self.model_name = model_name
        self.device = DEVICE
        
        print(f"Loading {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        # Apply freezing strategy
        if freeze_bert:
            self._freeze_layers(freeze_embeddings, unfreeze_last_n)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Print trainable parameter info
        self._print_trainable_params()
    
    def _freeze_layers(self, freeze_embeddings: bool = True, unfreeze_last_n: int = 2):
        """
        Strategic layer freezing for BERT.
        
        BERT architecture (bert-base-uncased):
        - embeddings (word, position, token_type)
        - encoder.layer.0 to encoder.layer.11 (12 layers)
        - pooler
        - classifier (added by AutoModelForSequenceClassification)
        
        Strategy: Freeze embeddings + first N layers, fine-tune last layers + classifier
        """
        # 1. Freeze embeddings
        if freeze_embeddings:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            print("✓ Frozen: Embeddings layer")
        
        # 2. Freeze all encoder layers first
        total_encoder_layers = len(self.model.bert.encoder.layer)  # 12 for bert-base
        
        for i, layer in enumerate(self.model.bert.encoder.layer):
            freeze_this = i < (total_encoder_layers - unfreeze_last_n)
            for param in layer.parameters():
                param.requires_grad = not freeze_this
            
            if freeze_this:
                print(f"  Frozen: Encoder layer {i}")
            else:
                print(f"  Trainable: Encoder layer {i}")
        
        # 3. Freeze pooler
        for param in self.model.bert.pooler.parameters():
            param.requires_grad = False
        print("✓ Frozen: Pooler layer")
        
        # 4. Classifier remains trainable by default
        print("✓ Trainable: Classification head")
    
    def _print_trainable_params(self):
        """Print summary of trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        frozen = total - trainable
        
        print(f"\n{'='*40}")
        print(f"BERT Parameter Summary:")
        print(f"  Total params:    {total:,}")
        print(f"  Trainable:       {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Frozen:          {frozen:,} ({frozen/total*100:.1f}%)")
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
    
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return predictions, true_labels
    
    def predict(self, texts: List[str], max_length: int = 512):
        """Make predictions on new texts."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**tokens)
                pred = torch.argmax(outputs.logits, dim=1)
                predictions.append(pred.item())
        
        return predictions