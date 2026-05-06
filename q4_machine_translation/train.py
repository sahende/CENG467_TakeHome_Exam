"""
Training script for Seq2Seq with Attention (Q4).
Fine-tunes T5, trains Seq2Seq, and compares both on Multi30k.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import sys
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, SEED, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODELS_DIR, RESULTS_DIR
from q4_machine_translation.preprocess import TranslationDataset, TranslationPreprocessor
from q4_machine_translation.seq2seq_attention import Encoder, Decoder, Seq2SeqAttention
from q4_machine_translation.transformer_model import TransformerMT
from q4_machine_translation.evaluate import TranslationEvaluator

torch.manual_seed(SEED)
np.random.seed(SEED)


class MTExperiment:
    """Machine Translation experiment comparing Seq2Seq+Attention vs Transformer."""
    
    def __init__(self):
        self.results = {}
        self.translations = {}
    
    # ================================================================
    # T5 FINE-TUNING
    # ================================================================
    def fine_tune_t5(self, epochs: int = 3):
        """Fine-tune T5-small on Multi30k with custom training loop."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        from datasets import load_dataset
        
        print("=" * 60)
        print("FINE-TUNING T5 on Multi30k")
        print("=" * 60)
        
        model_name = "t5-small"
        print(f"\nLoading {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        
        print("\nLoading Multi30k dataset...")
        dataset = load_dataset("bentrevett/multi30k")
        train_data = dataset['train'].select(range(2000))
        
        # Prepare data
        train_pairs = []
        for item in train_data:
            input_text = f"translate English to German: {item['en']}"
            train_pairs.append((input_text, item['de']))
        
        class MTDataset(Dataset):
            def __init__(self, pairs, tokenizer, max_len=128):
                self.pairs = pairs
                self.tokenizer = tokenizer
                self.max_len = max_len
            
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                src, tgt = self.pairs[idx]
                src_enc = self.tokenizer(src, max_length=self.max_len, truncation=True,
                                         padding='max_length', return_tensors='pt')
                tgt_enc = self.tokenizer(tgt, max_length=self.max_len, truncation=True,
                                         padding='max_length', return_tensors='pt')
                return {
                    'input_ids': src_enc['input_ids'].squeeze(),
                    'attention_mask': src_enc['attention_mask'].squeeze(),
                    'labels': tgt_enc['input_ids'].squeeze()
                }
        
        train_dataset = MTDataset(train_pairs, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        
        print(f"\nTraining for {epochs} epochs...")
        print("-" * 40)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
        save_path = os.path.join(MODELS_DIR, 't5_en_de_finetuned')
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\n✓ Fine-tuned T5 saved to {save_path}")
        
        return model, tokenizer
    
    # ================================================================
    # SEQ2SEQ TRAINING
    # ================================================================
    def train_seq2seq(self, epochs: int = 10) -> Tuple[Seq2SeqAttention, TranslationDataset]:
        """Train Seq2Seq with Attention on Multi30k."""
        
        print("=" * 60)
        print("TRAINING Seq2Seq + Attention")
        print("=" * 60)
        
        from datasets import load_dataset
        print("\nLoading Multi30k dataset...")
        dataset = load_dataset("bentrevett/multi30k")
        
        train_src = dataset['train']['en'][:2000]
        train_tgt = dataset['train']['de'][:2000]
        val_src = dataset['validation']['en'][:500]
        val_tgt = dataset['validation']['de'][:500]
        test_src = dataset['test']['en'][:500]
        test_tgt = dataset['test']['de'][:500]
        
        print(f"Train: {len(train_src)}, Val: {len(val_src)}, Test: {len(test_src)}")
        
        train_dataset = TranslationDataset(train_src, train_tgt, max_length=50, min_freq=2, build_vocab=True)
        val_dataset = TranslationDataset(val_src, val_tgt, src_vocab=train_dataset.src_vocab,
                                         tgt_vocab=train_dataset.tgt_vocab, max_length=50, build_vocab=False)
        test_dataset = TranslationDataset(test_src, test_tgt, src_vocab=train_dataset.src_vocab,
                                          tgt_vocab=train_dataset.tgt_vocab, max_length=50, build_vocab=False)
        
        src_vocab_size = len(train_dataset.src_vocab)
        tgt_vocab_size = len(train_dataset.tgt_vocab)
        print(f"Source vocab: {src_vocab_size}, Target vocab: {tgt_vocab_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=TranslationDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=TranslationDataset.collate_fn)
        
        encoder = Encoder(input_dim=src_vocab_size, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.5)
        decoder = Decoder(output_dim=tgt_vocab_size, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.5)
        model = Seq2SeqAttention(encoder, decoder, DEVICE).to(DEVICE)
        
        print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                src = batch['src'].to(DEVICE)
                tgt = batch['tgt'].to(DEVICE)
                
                optimizer.zero_grad()
                output = model(src, tgt, teacher_forcing_ratio=0.5)
                output = output[:, 1:, :].reshape(-1, tgt_vocab_size)
                tgt_out = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt_out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    src = batch['src'].to(DEVICE)
                    tgt = batch['tgt'].to(DEVICE)
                    output = model(src, tgt, teacher_forcing_ratio=0.0)
                    output = output[:, 1:, :].reshape(-1, tgt_vocab_size)
                    tgt_out = tgt[:, 1:].reshape(-1)
                    loss = criterion(output, tgt_out)
                    val_loss += loss.item()
            
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'seq2seq_best.pt'))
        
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'seq2seq_best.pt')))
        return model, test_dataset
    
    # ================================================================
    # TRANSLATION
    # ================================================================
    def translate_seq2seq(self, model, dataset, max_len=50):
        """Generate translations using trained Seq2Seq."""
        print("\nGenerating Seq2Seq translations...")
        model.eval()
        translations = []
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=TranslationDataset.collate_fn)
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(DEVICE)
                translated_ids = model.translate(src, max_len=max_len)
                for ids in translated_ids:
                    translations.append(dataset.decode_sentence(ids.tolist(), is_target=True))
        return translations
    
    def translate_transformer(self, texts):
        """Generate translations using fine-tuned T5."""
        print("\nGenerating Transformer (T5) translations...")
        finetuned_path = os.path.join(MODELS_DIR, 't5_en_de_finetuned')
        
        if os.path.exists(finetuned_path):
            print(f"Loading fine-tuned T5 from {finetuned_path}...")
            translator = TransformerMT(model_name=finetuned_path)
        else:
            print("Using pre-trained T5-small (not fine-tuned)...")
            translator = TransformerMT(model_name="t5-small")
        
        return translator.translate_batch(texts)
    
    # ================================================================
    # MAIN EXPERIMENT
    # ================================================================
    def run_experiment(self):
        """Run complete Q4 experiment."""
        
        # 0. Fine-tune T5 (if not already)
        finetuned_path = os.path.join(MODELS_DIR, 't5_en_de_finetuned')
        if not os.path.exists(finetuned_path):
            self.fine_tune_t5(epochs=3)
        else:
            print(f"✓ Fine-tuned T5 already exists at {finetuned_path}")
        
        # 1. Train Seq2Seq
        seq2seq_model, test_dataset = self.train_seq2seq(epochs=10)
        
        # 2. Test data
        print("\nPreparing test data...")
        test_src_texts = test_dataset.src_texts
        test_ref_texts = test_dataset.tgt_texts
        
        # 3. Generate
        seq2seq_translations = self.translate_seq2seq(seq2seq_model, test_dataset)
        transformer_translations = self.translate_transformer(test_src_texts)
        
        # 4. Evaluate
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        evaluator = TranslationEvaluator()
        comparison = evaluator.compare_models(test_ref_texts, seq2seq_translations, transformer_translations)
        
        # 5. Qualitative
        print(f"\n{'='*60}")
        print("QUALITATIVE EXAMPLES")
        print(f"{'='*60}")
        for i in range(3):
            print(f"\nExample {i+1}:")
            print(f"  Source:      {test_src_texts[i][:100]}")
            print(f"  Reference:   {test_ref_texts[i][:100]}")
            print(f"  Seq2Seq:     {seq2seq_translations[i][:100]}")
            print(f"  Transformer: {transformer_translations[i][:100]}")
        
        # 6. Save
        results = {'seq2seq': comparison['seq2seq'], 'transformer': comparison['transformer']}
        def convert(obj):
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        results_path = os.path.join(RESULTS_DIR, 'q4_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert(results), f, indent=4)
        print(f"\n✓ Results saved to {results_path}")
        print(f"\n{'='*60}")
        print("Q4 Complete!")
        print(f"{'='*60}")


def main():
    experiment = MTExperiment()
    experiment.run_experiment()


if __name__ == "__main__":
    main()