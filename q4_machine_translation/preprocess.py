"""
Preprocessing module for Machine Translation (Q4).
Handles tokenization and data preparation for Multi30k dataset.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple, Dict, Optional
import re


class TranslationDataset(Dataset):
    """
    Dataset for machine translation tasks.
    """
    
    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 src_vocab: Dict[str, int] = None, tgt_vocab: Dict[str, int] = None,
                 max_length: int = 50, min_freq: int = 2,
                 build_vocab: bool = True):
        """
        Initialize translation dataset.
        
        Args:
            src_texts: Source language texts
            tgt_texts: Target language texts
            src_vocab: Source vocabulary (if pre-built)
            tgt_vocab: Target vocabulary (if pre-built)
            max_length: Maximum sequence length
            min_freq: Minimum frequency for vocabulary
            build_vocab: Whether to build vocabulary
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_length = max_length
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, 
                               self.EOS_TOKEN, self.UNK_TOKEN]
        
        if build_vocab:
            self.src_vocab = self._build_vocabulary(src_texts, min_freq)
            self.tgt_vocab = self._build_vocabulary(tgt_texts, min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
        
        # Reverse vocabularies
        self.src_itos = {v: k for k, v in self.src_vocab.items()}
        self.tgt_itos = {v: k for k, v in self.tgt_vocab.items()}
    
    def _build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts
            min_freq: Minimum frequency threshold
            
        Returns:
            Word to index mapping
        """
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        # Filter by frequency and add special tokens
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        for token, count in counter.items():
            if count >= min_freq:
                vocab[token] = len(vocab)
        
        return vocab
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split
        text = text.lower()
        # Handle punctuation
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        tokens = text.split()
        return tokens
    
    def _numericalize(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """
        Convert text to indices.
        
        Args:
            text: Input text
            vocab: Vocabulary mapping
            
        Returns:
            List of token indices
        """
        tokens = self._tokenize(text)
        # Add SOS and EOS tokens
        indices = [vocab[self.SOS_TOKEN]]
        indices += [vocab.get(token, vocab[self.UNK_TOKEN]) for token in tokens]
        indices += [vocab[self.EOS_TOKEN]]
        
        # Truncate
        indices = indices[:self.max_length]
        
        return indices
    
    def __len__(self) -> int:
        return len(self.src_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        src_tokens = self._numericalize(self.src_texts[idx], self.src_vocab)
        tgt_tokens = self._numericalize(self.tgt_texts[idx], self.tgt_vocab)
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long)
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        Pads sequences within a batch.
        
        Args:
            batch: List of examples
            
        Returns:
            Batched tensors
        """
        src_batch = [item['src'] for item in batch]
        tgt_batch = [item['tgt'] for item in batch]
        
        # Pad sequences
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
        
        return {
            'src': src_padded,
            'tgt': tgt_padded
        }
    
    def decode_sentence(self, indices: List[int], is_target: bool = True) -> str:
        """
        Decode indices back to sentence.
        
        Args:
            indices: List of token indices
            is_target: Whether to use target vocabulary
            
        Returns:
            Decoded sentence
        """
        vocab = self.tgt_itos if is_target else self.src_itos
        
        tokens = []
        for idx in indices:
            if idx == 0:  # PAD
                continue
            token = vocab.get(idx, self.UNK_TOKEN)
            if token in [self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            tokens.append(token)
        
        return ' '.join(tokens)


class TranslationPreprocessor:
    """
    Preprocessor for Multi30k dataset.
    """
    
    def __init__(self, max_length: int = 50, min_freq: int = 2):
        """
        Initialize preprocessor.
        
        Args:
            max_length: Maximum sequence length
            min_freq: Minimum word frequency
        """
        self.max_length = max_length
        self.min_freq = min_freq
        self.dataset = None
    
    def load_multi30k(self) -> Dict:
        """
        Load and prepare Multi30k dataset.
        
        Returns:
            Dictionary with train, validation, and test datasets
        """
        from datasets import load_dataset
        
        print("Loading Multi30k dataset...")
        dataset = load_dataset("bentrevett/multi30k")
        
        return dataset
    
    def prepare_data(self, dataset) -> Tuple[TranslationDataset, ...]:
        """
        Prepare datasets for training.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Extract texts (English source, German target for example)
        train_src = dataset['train']['en']
        train_tgt = dataset['train']['de']
        val_src = dataset['validation']['en']
        val_tgt = dataset['validation']['de']
        test_src = dataset['test']['en']
        test_tgt = dataset['test']['de']
        
        # Create train dataset with vocabulary building
        train_dataset = TranslationDataset(
            train_src, train_tgt,
            max_length=self.max_length,
            min_freq=self.min_freq,
            build_vocab=True
        )
        
        # Create validation and test datasets using training vocab
        val_dataset = TranslationDataset(
            val_src, val_tgt,
            src_vocab=train_dataset.src_vocab,
            tgt_vocab=train_dataset.tgt_vocab,
            max_length=self.max_length,
            build_vocab=False
        )
        
        test_dataset = TranslationDataset(
            test_src, test_tgt,
            src_vocab=train_dataset.src_vocab,
            tgt_vocab=train_dataset.tgt_vocab,
            max_length=self.max_length,
            build_vocab=False
        )
        
        return train_dataset, val_dataset, test_dataset