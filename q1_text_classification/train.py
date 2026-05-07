"""
Training script for text classification models (Q1).
Trains and compares TF-IDF, BiLSTM, and BERT models.
Includes systematic preprocessing analysis.
All models trained under comparable conditions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import json
import copy
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, SEED, BATCH_SIZE, LEARNING_RATE, 
    NUM_EPOCHS, MAX_SEQ_LENGTH, MODELS_DIR, RESULTS_DIR
)
from q1_text_classification.preprocess import (
    TextPreprocessor, TFIDFVectorizer, prepare_dataset_for_training
)
from q1_text_classification.models import TFIDFClassifier, BiLSTMClassifier, BERTClassifier

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class TextClassificationExperiment:
    """
    Orchestrates the text classification experiment.
    Compares multiple models and preprocessing strategies under comparable conditions.
    """
    
    def __init__(self, data_dict: Dict):
        """
        Initialize experiment.
        
        Args:
            data_dict: Dictionary with 'train' and 'test' data
        """
        self.train_texts = data_dict['train']['texts']
        self.train_labels = data_dict['train']['labels']
        self.test_texts = data_dict['test']['texts']
        self.test_labels = data_dict['test']['labels']
        self.results = {}
        self.all_predictions = {}
        self.preprocessing_results = []
        
        # Split train into train/val for neural models (comparable condition: same split)
        self.train_texts_split, self.val_texts, self.train_labels_split, self.val_labels = \
            train_test_split(
                self.train_texts, self.train_labels, 
                test_size=0.2, random_state=SEED, stratify=self.train_labels
            )
    
    def run_preprocessing_analysis(self) -> List[Dict]:
        """
        Systematically analyze preprocessing decisions.
        Examines normalization, stopword removal, and truncation effects.
        Uses TF-IDF + Logistic Regression as baseline for fair comparison.
        
        Returns:
            List of dictionaries with preprocessing experiment results
        """
        print(f"\n{'='*70}")
        print("PREPROCESSING IMPACT ANALYSIS")
        print("Comparing: Normalization, Stopword Removal, Truncation Length")
        print(f"{'='*70}")
        
        # Define preprocessing configurations to test
        preprocessing_configs = [
            # (name, normalize, lowercase, remove_punct, remove_stop, max_length)
            ("1_Full_Preprocessing",      True,  True,  True,  True,  512),
            ("2_No_Stopword_Removal",     True,  True,  True,  False, 512),
            ("3_No_Normalization",        False, False, False, True,  512),
            ("4_Keep_Punctuation",        True,  True,  False, True,  512),
            ("5_Keep_Case",               True,  False, True,  True,  512),
            ("6_Truncation_128_tokens",   True,  True,  True,  True,  128),
            ("7_Truncation_256_tokens",   True,  True,  True,  True,  256),
            ("8_No_Preprocessing",        False, False, False, False, 512),
        ]
        
        for config_name, normalize, lowercase, remove_punct, remove_stop, max_len in preprocessing_configs:
            print(f"\n{'─'*50}")
            print(f"Configuration: {config_name}")
            print(f"  Normalize: {normalize} | Lowercase: {lowercase} | "
                  f"Remove Punct: {remove_punct} | Remove Stopwords: {remove_stop} | "
                  f"Max Length: {max_len}")
            
            # Create preprocessor with current configuration
            preprocessor = TextPreprocessor(
                tokenizer_type="basic", 
                max_length=max_len
            )
            
            # Process training texts with current config
            train_processed = []
            for text in self.train_texts:
                # Apply preprocessing steps manually for full control
                processed = text
                
                if normalize:
                    processed = preprocessor.normalize_text(
                        processed, 
                        lowercase=lowercase, 
                        remove_punctuation=remove_punct
                    )
                elif lowercase:
                    processed = processed.lower()
                
                # Tokenize
                tokens = preprocessor.tokenize_basic(processed)
                
                # Remove stopwords if configured
                if remove_stop:
                    tokens = preprocessor.remove_stopwords(tokens, remove=True)
                
                # Truncate
                tokens = tokens[:max_len]
                
                train_processed.append(' '.join(tokens))
            
            # Process test texts with SAME configuration
            test_processed = []
            for text in self.test_texts:
                processed = text
                
                if normalize:
                    processed = preprocessor.normalize_text(
                        processed, 
                        lowercase=lowercase, 
                        remove_punctuation=remove_punct
                    )
                elif lowercase:
                    processed = processed.lower()
                
                tokens = preprocessor.tokenize_basic(processed)
                
                if remove_stop:
                    tokens = preprocessor.remove_stopwords(tokens, remove=True)
                
                tokens = tokens[:max_len]
                
                test_processed.append(' '.join(tokens))
            
            # Vectorize with TF-IDF (same parameters for all configs)
            vectorizer = TFIDFVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train = vectorizer.fit_transform(train_processed)
            X_test = vectorizer.transform(test_processed)
            
            # Train Logistic Regression (comparable condition: same model)
            classifier = TFIDFClassifier(model_type="logistic_regression")
            classifier.train(X_train, self.train_labels)
            
            # Predict
            y_pred = classifier.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(self.test_labels, y_pred)
            f1_macro = f1_score(self.test_labels, y_pred, average='macro')
            f1_per_class = f1_score(self.test_labels, y_pred, average=None)
            
            result = {
                'configuration': config_name,
                'normalize': normalize,
                'lowercase': lowercase,
                'remove_punctuation': remove_punct,
                'remove_stopwords': remove_stop,
                'max_length': max_len,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_negative': f1_per_class[0] if len(f1_per_class) > 0 else 0,
                'f1_positive': f1_per_class[1] if len(f1_per_class) > 1 else 0,
                'vocab_size': len(vectorizer.vectorizer.vocabulary_),
                'num_features': X_train.shape[1]
            }
            
            self.preprocessing_results.append(result)
            
            print(f"   Accuracy: {accuracy:.4f} | Macro F1: {f1_macro:.4f} | "
                  f"Vocab Size: {result['vocab_size']}")
        
        # Save preprocessing analysis results
        pp_results_path = os.path.join(RESULTS_DIR, 'q1_preprocessing_analysis.json')
        with open(pp_results_path, 'w') as f:
            json.dump(self.preprocessing_results, f, indent=4)
        print(f"\n Preprocessing analysis saved to {pp_results_path}")
        
        # Print summary
        self._print_preprocessing_summary()
        
        return self.preprocessing_results
    
    def _print_preprocessing_summary(self):
        """Print preprocessing analysis summary."""
        print(f"\n{'='*70}")
        print("PREPROCESSING ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Configuration':<35} {'Accuracy':<12} {'F1 Macro':<12} {'Vocab Size':<12}")
        print(f"{'─'*70}")
        
        for r in self.preprocessing_results:
            print(f"{r['configuration']:<35} {r['accuracy']:<12.4f} "
                  f"{r['f1_macro']:<12.4f} {r['vocab_size']:<12}")
        
        # Best and worst
        best = max(self.preprocessing_results, key=lambda x: x['f1_macro'])
        worst = min(self.preprocessing_results, key=lambda x: x['f1_macro'])
        
        print(f"\n► Best: {best['configuration']} (F1={best['f1_macro']:.4f})")
        print(f"► Worst: {worst['configuration']} (F1={worst['f1_macro']:.4f})")
    
    def run_tfidf_experiment(self, model_type: str = "logistic_regression",
                             remove_stopwords: bool = True,
                             max_length: int = 512,
                             normalize: bool = True,
                             lowercase: bool = True,
                             remove_punctuation: bool = True) -> Tuple[Dict, np.ndarray]:
        """
        Run TF-IDF based model experiment with configurable preprocessing.
        
        Args:
            model_type: "logistic_regression" or "svm"
            remove_stopwords: Whether to remove stopwords
            max_length: Maximum sequence length
            normalize: Apply text normalization
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            
        Returns:
            Tuple of (metrics_dict, predictions_array)
        """
        model_name = f'TF-IDF + {model_type}'
        print(f"\n{'='*50}")
        print(f"Running {model_name} experiment")
        print(f"Preprocessing: normalize={normalize}, lowercase={lowercase}, "
              f"remove_punct={remove_punctuation}, remove_stop={remove_stopwords}, "
              f"max_len={max_length}")
        print(f"{'='*50}")
        
        # Create preprocessor
        preprocessor = TextPreprocessor(tokenizer_type="basic", max_length=max_length)
        
        # Process training texts
        train_processed = []
        for text in self.train_texts:
            if normalize:
                processed = preprocessor.normalize_text(
                    text, lowercase=lowercase, remove_punctuation=remove_punctuation
                )
            elif lowercase:
                processed = text.lower()
            else:
                processed = text
            
            tokens = preprocessor.tokenize_basic(processed)
            
            if remove_stopwords:
                tokens = preprocessor.remove_stopwords(tokens, remove=True)
            
            tokens = tokens[:max_length]
            train_processed.append(' '.join(tokens))
        
        # Process test texts (SAME configuration for comparable conditions)
        test_processed = []
        for text in self.test_texts:
            if normalize:
                processed = preprocessor.normalize_text(
                    text, lowercase=lowercase, remove_punctuation=remove_punctuation
                )
            elif lowercase:
                processed = text.lower()
            else:
                processed = text
            
            tokens = preprocessor.tokenize_basic(processed)
            
            if remove_stopwords:
                tokens = preprocessor.remove_stopwords(tokens, remove=True)
            
            tokens = tokens[:max_length]
            test_processed.append(' '.join(tokens))
        
        # Vectorize with TF-IDF
        vectorizer = TFIDFVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_processed)
        X_test = vectorizer.transform(test_processed)
        
        # Train classifier
        classifier = TFIDFClassifier(model_type=model_type)
        classifier.train(X_train, self.train_labels)
        
        # Get prediction probabilities (for error analysis)
        y_pred_proba = classifier.predict_proba(X_test)
        
        # Predict
        y_pred = classifier.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(self.test_labels, y_pred)
        f1_macro = f1_score(self.test_labels, y_pred, average='macro')
        
        key = f'tfidf_{model_type}'
        metrics = {
            'model': model_name,
            'model_type': 'TF-IDF (Sparse Representation)',
            'stopword_removal': remove_stopwords,
            'max_length': max_length,
            'normalize': normalize,
            'lowercase': lowercase,
            'remove_punctuation': remove_punctuation,
            'vocab_size': X_train.shape[1],
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'num_train_samples': len(self.train_labels),
            'num_test_samples': len(self.test_labels)
        }
        
        self.results[key] = metrics
        self.all_predictions[key] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': self.test_labels
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Vocabulary Size: {X_train.shape[1]}")
        
        return metrics, y_pred
    
    def run_bilstm_experiment(self, embedding_dim: int = 50,
                            hidden_dim: int = 64,
                            num_layers: int = 1,
                            max_length: int = 256) -> Dict:
        """Run BiLSTM model experiment."""
        
        print(f"\n{'='*50}")
        print("Running BiLSTM experiment ")
        print(f"Max Length: {max_length}, Embedding Dim: {embedding_dim}, Hidden Dim: {hidden_dim}")
        print(f"{'='*50}")
        
        # Build vocabulary - LIMIT vocab size
        preprocessor = TextPreprocessor(tokenizer_type="basic", max_length=max_length)
        
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        word_counts = {}
        idx = 2
        
        # Count words first
        for text in self.train_texts:
            processed = preprocessor.normalize_text(text, lowercase=True, remove_punctuation=True)
            tokens = processed.split()[:max_length]
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Only include frequent words (min_freq=3)
        MIN_FREQ = 3
        for token, count in word_counts.items():
            if count >= MIN_FREQ:
                word_to_idx[token] = idx
                idx += 1
        
        vocab_size = len(word_to_idx)
        print(f"Vocabulary size: {vocab_size} (min_freq={MIN_FREQ})")
        
        # Convert texts to sequences
        def texts_to_sequences(texts, word_to_idx, max_len):
            sequences = []
            for text in texts:
                processed = preprocessor.normalize_text(text, lowercase=True, remove_punctuation=True)
                tokens = processed.split()[:max_len]
                seq = [word_to_idx.get(t, 1) for t in tokens]  # 1 = <UNK>
                sequences.append(seq)
            return sequences
        
        train_sequences = texts_to_sequences(self.train_texts_split, word_to_idx, max_length)
        val_sequences = texts_to_sequences(self.val_texts, word_to_idx, max_length)
        test_sequences = texts_to_sequences(self.test_texts, word_to_idx, max_length)
        
        # Pad sequences
        train_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in train_sequences], 
            batch_first=True, padding_value=0
        )
        val_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in val_sequences], 
            batch_first=True, padding_value=0
        )
        test_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in test_sequences], 
            batch_first=True, padding_value=0
        )
        
        # Create data loaders
        train_dataset = TensorDataset(train_padded, torch.tensor(self.train_labels_split))
        val_dataset = TensorDataset(val_padded, torch.tensor(self.val_labels))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model 
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,    
            hidden_dim=hidden_dim,          
            num_layers=num_layers,          
            num_classes=2,
            dropout=0.7,
            pad_idx=0
        ).to(DEVICE)
        
        print(f"Model: BiLSTM | Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Training setup 
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05) 
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        # ============================================
        # BEST MODEL TRACKING 
        # ============================================
        best_val_acc = 0
        best_model_state = None
        best_epoch = 0
        patience = 4
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            
            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                f"Train Loss: {train_loss/len(train_loader):.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss/len(val_loader):.4f} - "
                f"Val Acc: {val_acc:.4f} - "
                )
            scheduler.step(avg_val_loss) 
            # ============================================
            # SAVE BEST MODEL 
            # ============================================
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Deep copy model state
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                print(f"  ✓ New best model! (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            
        # ============================================
        # LOAD BEST MODEL FOR TEST EVALUATION
        # ============================================
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")
        else:
            print("\n⚠ No best model saved, using last epoch")
        
        # Final test evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(test_padded.to(DEVICE))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
        
        accuracy = accuracy_score(self.test_labels, y_pred)
        f1_macro = f1_score(self.test_labels, y_pred, average='macro')
        
        from sklearn.metrics import classification_report
        print("\n" + classification_report(self.test_labels, y_pred, 
                                        target_names=['Negative', 'Positive']))
        
        metrics = {
            'model': 'BiLSTM',
            'model_type': 'BiLSTM (Dense Representation)',
            'vocab_size': vocab_size,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_acc,
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
        
        self.results['bilstm'] = metrics
        self.all_predictions['bilstm'] = {
            'predictions': y_pred,
            'probabilities': probs,
            'true_labels': self.test_labels
        }
        
        print(f"✓ Test Accuracy: {accuracy:.4f} | Test Macro F1: {f1_macro:.4f}")
        
        return metrics, y_pred
        
    
    def run_bert_experiment(self, model_name: str = "bert-base-uncased",
                            max_length: int = 256,
                            freeze_bert: bool = True,
                            unfreeze_last_n: int = 2) -> Dict:
        """
        Run BERT model experiment.
        
        Args:
            model_name: Pre-trained model name (e.g., "bert-base-uncased", "distilbert-base-uncased")
            max_length: Maximum sequence length
            freeze_bert: Whether to freeze BERT layers
            unfreeze_last_n: Number of last encoder layers to unfreeze (if freeze_bert=True)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*50}")
        print(f"Running BERT experiment with {model_name}")
        print(f"Max Length: {max_length} | Freeze: {freeze_bert} | Unfreeze layers: {unfreeze_last_n}")
        print(f"{'='*50}")
        
        # Initialize BERT classifier with freezing strategy
        bert_classifier = BERTClassifier(
            model_name=model_name, 
            num_classes=2,
            freeze_bert=freeze_bert,
            freeze_embeddings=True,
            unfreeze_last_n=unfreeze_last_n
        )
        
        # Create dataset class for BERT
        class BERTDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        # Create datasets using the SAME train/val/test split as other models
        train_dataset = BERTDataset(
            self.train_texts_split, self.train_labels_split, 
            bert_classifier.tokenizer, max_length
        )
        val_dataset = BERTDataset(
            self.val_texts, self.val_labels, 
            bert_classifier.tokenizer, max_length
        )
        test_dataset = BERTDataset(
            self.test_texts, self.test_labels, 
            bert_classifier.tokenizer, max_length
        )
        
        # Data loaders (smaller batch size for BERT due to memory)
        bert_batch_size = 16 if 'base' in model_name else 32  # DistilBERT can use larger batch
        train_loader = DataLoader(train_dataset, batch_size=bert_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bert_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=bert_batch_size)
        
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
            f"Test batches: {len(test_loader)}")
        
        # Optimizer - only optimize trainable parameters (frozen ones excluded)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, bert_classifier.model.parameters()), 
            lr=2e-4,  # Higher LR for frozen BERT
            weight_decay=0.01
        )
        
        # Learning rate scheduler - linear decay
        num_bert_epochs = 5
        total_steps = len(train_loader) * num_bert_epochs
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=total_steps
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        print(f"\nTraining BERT for {num_bert_epochs} epochs...")
        print("-" * 60)
        
        for epoch in range(num_bert_epochs):
            # Train one epoch
            train_loss = bert_classifier.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_preds, val_true = bert_classifier.evaluate(val_loader)
            val_acc = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{num_bert_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Acc: {val_acc:.4f} - "
                f"Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() for k, v in bert_classifier.model.state_dict().items()
                }
                print(f"  ✓ New best model! (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for evaluation
        if best_model_state is not None:
            bert_classifier.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model (Val Acc: {best_val_acc:.4f})")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        y_pred, y_true = bert_classifier.evaluate(test_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Detailed classification report
        from sklearn.metrics import classification_report
        print("\n" + classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
        
        # Store results
        key = 'bert'
        # run_bert_experiment içinde metrics'i düzelt:
        metrics = {
            'model': f'BERT ({model_name})',
            'model_type': 'BERT (Contextual Representation)',
            'accuracy': accuracy,          # ← test_accuracy değil, accuracy olacak
            'f1_macro': f1_macro,          # ← test_f1_macro değil, f1_macro olacak
            'freeze_bert': freeze_bert,
            'unfreeze_last_n': unfreeze_last_n,
            'max_length': max_length,
            'batch_size': bert_batch_size,
            'epochs_trained': epoch + 1,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': accuracy,     # ← ekstra detay olarak kalabilir
            'test_f1_macro': f1_macro,     # ← ekstra detay olarak kalabilir
            'num_train_samples': len(self.train_labels_split),
            'num_val_samples': len(self.val_labels),
            'num_test_samples': len(self.test_labels)
        }
        
        self.results[key] = metrics
        self.all_predictions[key] = {
            'predictions': y_pred,
            'probabilities': None,  # BERT evaluate doesn't return probs in current setup
            'true_labels': y_true
        }
        
        print(f"\n{'='*50}")
        print(f"BERT Results:")
        print(f"  Test Accuracy:  {accuracy:.4f}")
        print(f"  Test Macro F1:  {f1_macro:.4f}")
        print(f"  Best Val Acc:   {best_val_acc:.4f}")
        print(f"{'='*50}")
        
        return metrics, y_pred
    
    def run_all_experiments(self, run_preprocessing_analysis: bool = True,
                           run_tfidf: bool = True,
                           run_bilstm: bool = True,
                           run_bert: bool = True):
        """
        Run all experiments in sequence.
        
        Args:
            run_preprocessing_analysis: Run preprocessing impact analysis
            run_tfidf: Run TF-IDF experiment
            run_bilstm: Run BiLSTM experiment
            run_bert: Run BERT experiment
        """
        print(f"\n{'#'*70}")
        print(f"TEXT CLASSIFICATION EXPERIMENTS")
        print(f"Device: {DEVICE} | Seed: {SEED}")
        print(f"Train samples: {len(self.train_labels)} | Test samples: {len(self.test_labels)}")
        print(f"{'#'*70}")
        
        # 1. Preprocessing Analysis (always run first)
        if run_preprocessing_analysis:
            self.run_preprocessing_analysis()
        
        # 2. TF-IDF + Logistic Regression (with best preprocessing from analysis)
        if run_tfidf:
            self.run_tfidf_experiment(
                model_type="logistic_regression",
                remove_stopwords=True,
                normalize=True,
                lowercase=True,
                remove_punctuation=True
            )
        
        # 3. BiLSTM (Neural Model)
        if run_bilstm:
            self.run_bilstm_experiment()
        
        # 4. BERT (Transformer Model)
        if run_bert:
            self.run_bert_experiment(model_name="bert-base-uncased")
        
        # Save all results
        self.save_results()
        
        # Print final comparison
        self.print_model_comparison()
    
    def print_model_comparison(self):
        """Print comparison table of all models."""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Accuracy':<12} {'F1 Macro':<12} {'Type':<25}")
        print(f"{'─'*70}")
        
        for key, metrics in self.results.items():
            if 'accuracy' in metrics:
                model_name = metrics.get('model', key)
                model_type = metrics.get('model_type', 'N/A')
                print(f"{model_name:<30} {metrics['accuracy']:<12.4f} "
                      f"{metrics['f1_macro']:<12.4f} {model_type:<25}")
        
        print(f"\n All results saved to {os.path.join(RESULTS_DIR, 'q1_results.json')}")
    
    def save_results(self):
        """Save all results to JSON file."""
        import numpy as np
        
        # Helper function to convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Save model results
        results_path = os.path.join(RESULTS_DIR, 'q1_results.json')
        serializable_results = convert_to_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\n Model results saved to {results_path}")
        
        # Save predictions for error analysis
        predictions_path = os.path.join(RESULTS_DIR, 'q1_predictions.json')
        preds_serializable = {}
        
        for key, value in self.all_predictions.items():
            predictions = value['predictions']
            true_labels = value['true_labels']
            
            # Convert to plain Python lists
            if hasattr(predictions, 'tolist'):
                predictions = predictions.tolist()
            elif isinstance(predictions, np.ndarray):
                predictions = list(predictions)
            else:
                predictions = list(predictions)
            
            if hasattr(true_labels, 'tolist'):
                true_labels = true_labels.tolist()
            elif isinstance(true_labels, np.ndarray):
                true_labels = list(true_labels)
            else:
                true_labels = list(true_labels)
            
            # Convert any remaining numpy scalars in lists
            predictions = [int(p) if hasattr(p, 'item') else p for p in predictions]
            true_labels = [int(t) if hasattr(t, 'item') else t for t in true_labels]
            
            preds_serializable[key] = {
                'predictions': predictions,
                'true_labels': true_labels,
                'has_probabilities': value.get('probabilities') is not None
            }
        
        with open(predictions_path, 'w') as f:
            json.dump(preds_serializable, f, indent=4)
        print(f" Predictions saved to {predictions_path}")
    
    def get_predictions_for_analysis(self) -> Dict:
        """
        Get all predictions for error analysis.
        
        Returns:
            Dictionary with model predictions for analysis module
        """
        return self.all_predictions


def main():
    """Main execution function for Q1."""
    from datasets import load_dataset
    
    print("Loading IMDB dataset...")
    
    # Load full dataset
    dataset = load_dataset("imdb")
    
    # IMDB is ordered (all negatives first, then positives)
    # must shuffle before selecting subset to get balanced classes
    train_data = dataset['train'].shuffle(seed=SEED).select(range(2000))
    test_data = dataset['test'].shuffle(seed=SEED).select(range(500))
    
    # Prepare data
    data_dict = {
        'train': {
            'texts': train_data['text'],
            'labels': train_data['label']
        },
        'test': {
            'texts': test_data['text'],
            'labels': test_data['label']
        }
    }
    
    # Verify class distribution
    train_labels = np.array(data_dict['train']['labels'])
    test_labels = np.array(data_dict['test']['labels'])
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}")  # Should show ~1000 each
    print(f"Class distribution - Test: {np.bincount(test_labels)}")    # Should show ~250 each
    
    # Safety check
    if len(np.unique(train_labels)) < 2:
        print("ERROR: Only one class in training data! Shuffling may have failed.")
        return
    if len(np.unique(test_labels)) < 2:
        print("ERROR: Only one class in test data! Shuffling may have failed.")
        return
    
    print(f"✓ Classes balanced. Ready for experiments.\n")
    
    # Create experiment
    experiment = TextClassificationExperiment(data_dict)
    
    # Run all experiments
    experiment.run_all_experiments(
        run_preprocessing_analysis=True,   # Preprocessing impact analysis
        run_tfidf=True,                    # TF-IDF + Logistic Regression
        run_bilstm=True,                   # BiLSTM 
        run_bert=True                      # BERT
    )
    
    print(f"\n{'='*70}")
    print("Q1 Experiments Complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()