"""
Training script for Named Entity Recognition (Q2).
Compares CRF, and BERT-based approaches.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, SEED, BATCH_SIZE, LEARNING_RATE, 
    NUM_EPOCHS, MODELS_DIR, RESULTS_DIR
)
from q2_ner.preprocess import BIOTagger, CoNLLPreprocessor
from q2_ner.models import  NERBERTModel


class NERExperiment:
    """
    Orchestrates NER experiments comparing different models.
    """
    
    def __init__(self, dataset):
        """
        Initialize experiment.
        
        Args:
            dataset: CoNLL-2003 dataset
        """
        self.dataset = dataset
        self.bio_tagger = BIOTagger()
        self.preprocessor = CoNLLPreprocessor()
        self.results = {}
        self.all_predictions = {}
        
        # BIO etiketlerini oluştur (dataset'te yok, biz üretiyoruz)
        print("Preparing BIO tags from NER tags...")
        self._prepare_bio_tags()
    
    def _prepare_bio_tags(self):
        """Convert numerical NER tags to BIO string format."""
        for split in ['train', 'validation', 'test']:
            bio_tags_list = []
            for ner_tags in self.dataset[split]['ner_tags']:
                bio = [self.bio_tagger.id_to_tag.get(t, 'O') for t in ner_tags]
                bio_tags_list.append(bio)
            # Dataset'e yeni sütun olarak ekle
            self.dataset[split] = self.dataset[split].add_column('bio_tags', bio_tags_list)
        
        print(f"✓ BIO tags created for all splits")
        print(f"  Train: {len(self.dataset['train'])} samples")
        print(f"  Validation: {len(self.dataset['validation'])} samples")
        print(f"  Test: {len(self.dataset['test'])} samples")
    
    def run_crf_experiment(self) -> Dict:
        """
        Run CRF model experiment using sklearn-crfsuite.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*50)
        print("Running CRF Experiment")
        print("="*50)
        
        import sklearn_crfsuite
        
        # Get tokens and BIO tags
        train_tokens = self.dataset['train']['tokens']
        train_bio = self.dataset['train']['bio_tags']
        test_tokens = self.dataset['test']['tokens']
        test_bio = self.dataset['test']['bio_tags']
        
        # Prepare features
        X_train, y_train = self.preprocessor.prepare_data_for_crf(train_tokens, train_bio)
        X_test, y_test = self.preprocessor.prepare_data_for_crf(test_tokens, test_bio)
        
        # Train CRF
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        print("Training CRF model...")
        crf.fit(X_train, y_train)
        
        # Predict
        print("Making predictions...")
        y_pred = crf.predict(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nCRF Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        metrics = {
            'model': 'CRF (Classical)',
            'model_type': 'Classical ML',
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.results['crf'] = metrics
        self.all_predictions['crf'] = {
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        return metrics, y_pred, y_test
    


    
    def run_bert_ner_experiment(self, model_name: str = "bert-base-uncased",
                                epochs: int = 3,
                                freeze_bert: bool = True,
                                unfreeze_last_n: int = 2) -> Dict:
        """Run BERT-based NER experiment."""
        
        print("\n" + "="*50)
        print(f"Running BERT NER Experiment with {model_name}")
        print(f"Freeze BERT: {freeze_bert} | Unfreeze last {unfreeze_last_n} layers")
        print("="*50)
        
        # Initialize BERT NER model with freezing
        ner_bert = NERBERTModel(
            model_name=model_name,
            num_labels=self.bio_tagger.num_tags,
            freeze_bert=freeze_bert,
            freeze_embeddings=True,
            unfreeze_last_n=unfreeze_last_n
        )
        
        # Tokenize and align labels
        print("Tokenizing with BERT tokenizer...")
        
        def tokenize_and_align(examples):
            tokenized_inputs = ner_bert.tokenizer(
                examples['tokens'],
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            labels = []
            for i, word_ids in enumerate([tokenized_inputs.word_ids(batch_index=j) 
                                          for j in range(len(examples['tokens']))]):
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)  # Special tokens
                    elif word_idx != previous_word_idx:
                        # First token of word
                        bio_tag = examples['bio_tags'][i][word_idx]
                        label_ids.append(self.bio_tagger.tag_to_id.get(bio_tag, 0))
                    else:
                        label_ids.append(-100)  # Subword tokens
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs['labels'] = torch.tensor(labels)
            return tokenized_inputs
        
        # Create simple dataset class
        class NERDataset(Dataset):
            def __init__(self, split_data):
                self.tokens = split_data['tokens']
                self.bio_tags = split_data['bio_tags']
            
            def __len__(self):
                return len(self.tokens)
            
            def __getitem__(self, idx):
                return {
                    'tokens': self.tokens[idx],
                    'bio_tags': self.bio_tags[idx]
                }
        
        train_data = NERDataset(self.dataset['train'])
        test_data = NERDataset(self.dataset['test'])
        
        # Custom collate function
        def collate_fn(batch):
            tokens_list = [item['tokens'] for item in batch]
            bio_list = [item['bio_tags'] for item in batch]
            
            encoding = ner_bert.tokenizer(
                tokens_list,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Align labels
            labels = []
            for i, word_ids in enumerate([encoding.word_ids(batch_index=j) for j in range(len(tokens_list))]):
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        bio_tag = bio_list[i][word_idx]
                        label_ids.append(self.bio_tagger.tag_to_id.get(bio_tag, 0))
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            encoding['labels'] = torch.tensor(labels)
            return encoding
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=16, collate_fn=collate_fn)
        
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # Training
        optimizer = optim.AdamW(ner_bert.model.parameters(), lr=2e-5)
        
        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss = ner_bert.train_epoch(train_loader, optimizer)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        print("\nEvaluating on test set...")
        y_pred, y_true = ner_bert.evaluate(test_loader, self.bio_tagger.id_to_tag)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nBERT NER Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        metrics = {
            'model': f'BERT NER ({model_name})',
            'model_type': 'Transformer (Contextual)',
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.results['bert_ner'] = metrics
        self.all_predictions['bert_ner'] = {
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        return metrics, y_pred, y_true
    
    def print_comparison(self):
        """Print comparison of all models."""
        print(f"\n{'='*60}")
        print("NER MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"{'─'*60}")
        
        for key, metrics in self.results.items():
            print(f"{metrics['model']:<30} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
    
    def save_results(self):
        """Save all results to JSON."""
        import numpy as np
        
        def convert(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert(i) for i in obj]
            elif isinstance(obj, tuple): return [convert(i) for i in obj]
            return obj
        
        # 1. Save model metrics
        results_path = os.path.join(RESULTS_DIR, 'q2_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert(self.results), f, indent=4)
        print(f"✓ Results saved to {results_path}")
        
        # 2. Save predictions for error analysis (BU EKSİKTİ!)
        predictions_path = os.path.join(RESULTS_DIR, 'q2_predictions.json')
        preds_to_save = {}
        for key, value in self.all_predictions.items():
            preds_to_save[key] = {
                'predictions': value['predictions'],
                'true_labels': value['true_labels']
            }
        
        with open(predictions_path, 'w') as f:
            json.dump(convert(preds_to_save), f, indent=4)
        print(f"✓ Predictions saved to {predictions_path}")


def main():
    """Main execution for Q2."""
    from datasets import load_dataset
    
    print("Loading CoNLL-2003 dataset...")
    dataset = load_dataset("lhoestq/conll2003")
    
    # Create experiment
    experiment = NERExperiment(dataset)
    
    # Run CRF experiment
    crf_metrics, crf_preds, crf_true = experiment.run_crf_experiment()

    # Run BERT NER experiment
    bert_metrics, bert_preds, bert_true = experiment.run_bert_ner_experiment(epochs=3)
    
    # Save results
    experiment.save_results()
    
    # Print comparison
    experiment.print_comparison()


if __name__ == "__main__":
    main()