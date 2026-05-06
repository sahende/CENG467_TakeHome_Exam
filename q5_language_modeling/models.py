"""
Model implementations for language modeling (Q5).
Includes N-gram and GPT-2 (transformer) models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class NGramModel:
    """
    N-gram language model with optional smoothing.
    """
    
    def __init__(self, n: int = 3, smoothing: str = 'laplace', 
                 alpha: float = 1.0):
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
    
    def _get_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, corpus: List[List[str]]):
        """Train the N-gram model."""
        for sentence in corpus:
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            self.vocab.update(tokens)
            ngrams = self._get_ngrams(tokens)
            for ngram in ngrams:
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        self.vocab_size = len(self.vocab)
    
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Calculate probability of word given context."""
        context_count = self.context_counts.get(context, 0)
        word_count = self.ngram_counts.get(context, {}).get(word, 0)
        
        if self.smoothing == 'laplace':
            return (word_count + self.alpha) / (context_count + self.alpha * self.vocab_size)
        elif self.smoothing == 'none':
            if context_count == 0:
                return 1.0 / self.vocab_size
            return word_count / context_count
        return 1.0 / self.vocab_size
    
    def perplexity(self, test_corpus: List[List[str]]) -> float:
        """Calculate perplexity on test data."""
        total_log_prob = 0
        total_words = 0
        
        for sentence in test_corpus:
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i-self.n+1:i])
                word = tokens[i]
                prob = self.probability(word, context)
                total_log_prob += math.log2(max(prob, 1e-10))
                total_words += 1
        
        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        return 2 ** (-avg_log_prob)
    
    def generate(self, seed: List[str], max_length: int = 20) -> List[str]:
        """Generate text using the model."""
        if len(seed) < self.n - 1:
            seed = ['<s>'] * (self.n - 1 - len(seed)) + seed
        
        generated = seed.copy()
        
        for _ in range(max_length):
            context = tuple(generated[-(self.n-1):])
            if context not in self.ngram_counts:
                break
            
            words = list(self.ngram_counts[context].keys())
            counts = list(self.ngram_counts[context].values())
            probs = np.array(counts) / sum(counts)
            next_word = np.random.choice(words, p=probs)
            
            if next_word == '</s>':
                break
            generated.append(next_word)
        
        return generated


class GPT2Model:
    """
    GPT-2 based language model (transformer).
    Uses pre-trained GPT-2 for perplexity and text generation.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_name}...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def perplexity(self, texts: List[str], batch_size: int = 4) -> float:
        """Calculate perplexity on test texts."""
        from torch.utils.data import DataLoader, Dataset
        
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len=256):
                self.texts = [t for t in texts if t.strip()]
                self.tokenizer = tokenizer
                self.max_len = max_len
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx], max_length=self.max_len, 
                    truncation=True, padding='max_length', return_tensors='pt'
                )
                return {
                    'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze()
                }
        
        dataset = TextDataset(texts, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item() * attention_mask.sum().item()
                total_tokens += attention_mask.sum().item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return math.exp(avg_loss)
    
    def generate(self, seed_text: str, max_length: int = 50, 
                 temperature: float = 0.8) -> str:
        """Generate text from seed."""
        input_ids = self.tokenizer.encode(seed_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)