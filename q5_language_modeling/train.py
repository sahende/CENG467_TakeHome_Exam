"""
Training script for language modeling (Q5).
Compares N-gram and GPT-2 (transformer) models on WikiText-2.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, SEED, BATCH_SIZE, NUM_EPOCHS, MODELS_DIR, RESULTS_DIR
)
from q5_language_modeling.models import NGramModel, GPT2Model

torch.manual_seed(SEED)
np.random.seed(SEED)


class LMExperiment:
    """Language modeling experiment comparing N-gram and GPT-2."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {}
        self.generated_samples = {}
        
        self.train_texts = [t['text'] for t in dataset['train'] if t['text'].strip()]
        self.test_texts = [t['text'] for t in dataset['test'] if t['text'].strip()]
    
    # ================================================================
    # N-GRAM EXPERIMENT
    # ================================================================
    def run_ngram_experiment(self, n: int = 3, smoothing: str = 'laplace') -> Dict:
        """Run N-gram experiment."""
        print(f"\n{'='*50}")
        print(f"Running N-gram ({n}-gram) with {smoothing} smoothing")
        print(f"{'='*50}")
        
        train_tokens = [t.lower().split() for t in self.train_texts]
        test_tokens = [t.lower().split() for t in self.test_texts]
        
        model = NGramModel(n=n, smoothing=smoothing)
        model.train(train_tokens)
        
        test_ppl = model.perplexity(test_tokens)
        
        # ================================================================
        # SEED SELECTION FOR TEXT GENERATION
        # ================================================================
        # We use a mixed selection strategy:
        # - 3 generic seeds: hand-picked to demonstrate model capability on common patterns
        # - 3 test-set seeds: randomly sampled to evaluate performance on real Wikipedia text
        # This provides a balanced evaluation of both idealized and realistic generation quality.

        import random
        random.seed(SEED)  # Reproducible selection across runs

        # Generic seeds - common English patterns that any good LM should continue fluently
        seeds = [
            "The company announced",         # Business/news pattern
            "In recent years , the",         # Temporal/general pattern
            "Scientists have discovered"     # Scientific pattern
        ]
        references = [
            "[Generic prompt - no reference]",
            "[Generic prompt - no reference]",
            "[Generic prompt - no reference]"
        ]

        # Test-set seeds - drawn from WikiText-2 to assess real-world performance
        test_sentences = []
        for text in self.test_texts:
            sentences = text.split('.')
            for sent in sentences:
                words = sent.strip().split()
                if len(words) >= 5:  # Minimum 5 words for meaningful context
                    test_sentences.append(words)

        # Randomly sample 3 sentences from test set
        selected = random.sample(test_sentences, min(3, len(test_sentences)))

        for words in selected:
            seed = ' '.join(words[:3])           # First 3 words as seed
            reference = ' '.join(words)          # Full sentence as reference
            seeds.append(seed)
            references.append(reference)

        print(f"  Seeds for generation: {seeds}")
        
        generated_samples = []
        for i, seed_text in enumerate(seeds):
            seed_tokens = seed_text.lower().split()
            sample = model.generate(seed_tokens, max_length=30)
            generated = ' '.join(sample)
            generated_samples.append(f"Seed: {seed_text}\n  Reference: {references[i][:100]}...\n  → {generated}")
        
        generated_text = "\n".join(generated_samples)
        
        metrics = {
            'model': f'{n}-gram ({smoothing})',
            'test_perplexity': test_ppl,
            'generated_sample': generated_text,
            'vocab_size': model.vocab_size
        }
        
        self.results['ngram'] = metrics
        self.generated_samples['ngram'] = generated_text
        
        print(f"  Test Perplexity: {test_ppl:.2f}")
        print(f"  Vocab Size: {model.vocab_size}")
        print(f"  Generated samples:")
        for gs in generated_samples:
            print(f"    {gs[:150]}...")
        
        return metrics
    
    # ================================================================
    # GPT-2 EXPERIMENT 
    # ================================================================
    def run_gpt2_experiment(self, model_name: str = "gpt2", 
                            fine_tune: bool = True,
                            fine_tune_epochs: int = 3,
                            fine_tune_samples: int = 200) -> Dict:
        """Run GPT-2 experiment."""
        print(f"\n{'='*50}")
        print(f"Running GPT-2 ({model_name})")
        if fine_tune:
            print(f"Fine-tuning on {fine_tune_samples} samples for {fine_tune_epochs} epoch(s)")
        print(f"{'='*50}")
        
        # Load model
        model = GPT2Model(model_name=model_name)
        
        # Fine-tune on training data
        if fine_tune:
            import torch.optim as optim
            from torch.optim import AdamW 
            
            # Prepare training data
            train_subset = [t for t in self.train_texts if len(t.split()) > 10][:fine_tune_samples]
            train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
            
            optimizer = AdamW(model.model.parameters(), lr=5e-5)
            model.model.train()
            
            for epoch in range(fine_tune_epochs):
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    enc = model.tokenizer(
                        batch, max_length=128, truncation=True,
                        padding='max_length', return_tensors='pt'
                    )
                    input_ids = enc['input_ids'].to(model.device)
                    attention_mask = enc['attention_mask'].to(model.device)
                    
                    optimizer.zero_grad()
                    outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"  Epoch {epoch+1}/{fine_tune_epochs} - Loss: {avg_loss:.4f}")
            
            # Save fine-tuned model
            save_path = os.path.join(MODELS_DIR, 'gpt2_wikitext')
            model.model.save_pretrained(save_path)
            model.tokenizer.save_pretrained(save_path)
            print(f"   Fine-tuned model saved to {save_path}")
            model.model.eval()
        
        # Perplexity
        test_subset = self.test_texts[:50]
        print(f"  Calculating perplexity on {len(test_subset)} samples...")
        test_ppl = model.perplexity(test_subset)
        
        
        # ================================================================
        # SEED SELECTION FOR TEXT GENERATION
        # ================================================================
        # We use a mixed selection strategy:
        # - 3 generic seeds: hand-picked to demonstrate model capability on common patterns
        # - 3 test-set seeds: randomly sampled to evaluate performance on real Wikipedia text
        # This provides a balanced evaluation of both idealized and realistic generation quality.

        import random
        random.seed(SEED)  # Reproducible selection across runs

        # Generic seeds - common English patterns that any good LM should continue fluently
        seeds = [
            "The company announced",         # Business/news pattern
            "In recent years , the",         # Temporal/general pattern
            "Scientists have discovered"     # Scientific pattern
        ]
        references = [
            "[Generic prompt - no reference]",
            "[Generic prompt - no reference]",
            "[Generic prompt - no reference]"
        ]

        # Test-set seeds - drawn from WikiText-2 to assess real-world performance
        test_sentences = []
        for text in self.test_texts:
            sentences = text.split('.')
            for sent in sentences:
                words = sent.strip().split()
                if len(words) >= 5:  # Minimum 5 words for meaningful context
                    test_sentences.append(words)

        # Randomly sample 3 sentences from test set
        selected = random.sample(test_sentences, min(3, len(test_sentences)))

        for words in selected:
            seed = ' '.join(words[:3])           # First 3 words as seed
            reference = ' '.join(words)          # Full sentence as reference
            seeds.append(seed)
            references.append(reference)

        print(f"  Seeds for generation: {seeds}")
        
        generated_samples = []
        for i, seed in enumerate(seeds):
            gen = model.generate(seed, max_length=40, temperature=0.8)
            generated_samples.append(f"Seed: {seed}\n  Reference: {references[i][:100]}...\n  → {gen}")
        
        generated_text = "\n".join(generated_samples)
        
        suffix = ' fine-tuned' if fine_tune else ' (zero-shot)'
        metrics = {
            'model': f'GPT-2 ({model_name}){suffix}',
            'test_perplexity': test_ppl,
            'generated_sample': generated_text,
            'vocab_size': 50257
        }
        
        self.results['gpt2'] = metrics
        self.generated_samples['gpt2'] = generated_text
        
        print(f"  Test Perplexity: {test_ppl:.2f}")
        print(f"  Generated samples:")
        for gs in generated_samples:
            print(f"    {gs[:150]}...")
        
        return metrics
    
    # ================================================================
    # RESULTS
    # ================================================================
    def print_comparison(self):
        """Print comparison table."""
        print(f"\n{'='*70}")
        print("LANGUAGE MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Test PPL':<15} {'Vocab Size':<15}")
        print(f"{'─'*60}")
        
        for key in ['ngram', 'gpt2']:
            if key in self.results:
                r = self.results[key]
                print(f"{r['model']:<30} {r['test_perplexity']:<15.2f} {r['vocab_size']:<15,}")
        
        # Best model
        valid = {k: v for k, v in self.results.items() if v['test_perplexity'] < 1e6}
        if valid:
            best = min(valid.values(), key=lambda x: x['test_perplexity'])
            print(f"\n Best Perplexity: {best['model']} (PPL={best['test_perplexity']:.2f})")
        
        print(f"\n Note: Lower perplexity = better, but generation quality also matters.")
    
    def save_results(self):
        """Save results to JSON and samples to TXT."""
        def convert(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        # Save metrics
        results_path = os.path.join(RESULTS_DIR, 'q5_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert(self.results), f, indent=4)
        print(f"\n✓ Results saved to {results_path}")
        
        # Save generated samples
        samples_path = os.path.join(RESULTS_DIR, 'q5_generated_samples.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Q5 - LANGUAGE MODELING: GENERATED TEXT SAMPLES\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, sample in self.generated_samples.items():
                f.write(f"{'─'*60}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'─'*60}\n")
                f.write(sample + "\n\n")
        print(f"✓ Generated samples saved to {samples_path}")


def main():
    """Main execution for Q5."""
    from datasets import load_dataset
    
    print("=" * 60)
    print("Q5 - LANGUAGE MODELING")
    print("=" * 60)
    
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Use subset for quick experimentation
    dataset_small = {
        'train': dataset['train'].select(range(500)),
        'test': dataset['test'].select(range(100))
    }
    
    print(f"Train: {len(dataset_small['train'])} articles")
    print(f"Test:  {len(dataset_small['test'])} articles")
    
    experiment = LMExperiment(dataset_small)
    
    # Run experiments
    experiment.run_ngram_experiment(n=3, smoothing='laplace')
    experiment.run_gpt2_experiment(
        model_name="gpt2", 
        fine_tune=True, 
        fine_tune_epochs=3,
        fine_tune_samples=200
    )
    
    # Compare
    experiment.print_comparison()
    
    # Save
    experiment.save_results()
    
    print(f"\n{'='*60}")
    print("Q5 Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()