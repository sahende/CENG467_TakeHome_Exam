"""
Preprocessing module for Named Entity Recognition (Q2).
Implements BIO tagging and data preparation for CoNLL-2003.
"""
import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED

np.random.seed(SEED)


class BIOTagger:
    """
    Implements BIO (Begin, Inside, Outside) tagging scheme for NER.
    """
    
    # Standard CoNLL-2003 entity types
    ENTITY_TYPES = ['PER', 'ORG', 'LOC', 'MISC']
    
    def __init__(self):
        """Initialize BIO tagger with label mappings."""
        # Create BIO tag vocabulary
        self.tags = ['O']  # Outside tag
        for entity in self.ENTITY_TYPES:
            self.tags.append(f'B-{entity}')  # Begin tag
            self.tags.append(f'I-{entity}')  # Inside tag
        
        # Create mappings
        self.tag_to_id = {tag: i for i, tag in enumerate(self.tags)}
        self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}
        self.num_tags = len(self.tags)
    
    def encode_labels(self, labels: List[str]) -> List[int]:
        """Convert BIO labels to integer IDs."""
        return [self.tag_to_id.get(label, self.tag_to_id['O']) for label in labels]
    
    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """Convert integer IDs back to BIO labels."""
        return [self.id_to_tag.get(id, 'O') for id in label_ids]
    
    @staticmethod
    def convert_to_bio(words: List[str], entities: List[Tuple[int, int, str]]) -> List[str]:
        """
        Convert entity annotations to BIO format.
        
        Args:
            words: List of tokens
            entities: List of (start, end, entity_type) tuples
            
        Returns:
            List of BIO tags aligned with words
        """
        tags = ['O'] * len(words)
        
        for start, end, entity_type in entities:
            if start < len(words) and end <= len(words):
                tags[start] = f'B-{entity_type}'
                for i in range(start + 1, end):
                    tags[i] = f'I-{entity_type}'
        
        return tags
    
    def validate_alignment(self, tokens: List[str], labels: List[str]) -> bool:
        """
        Validate that tokens and labels are properly aligned.
        
        Args:
            tokens: List of tokens
            labels: List of BIO labels
            
        Returns:
            True if alignment is valid
        """
        if len(tokens) != len(labels):
            print(f"Warning: Token count ({len(tokens)}) != Label count ({len(labels)})")
            return False
        
        # Check BIO consistency
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                # Previous label should be B- or I- of same type
                if i == 0:
                    print(f"Warning: Sequence starts with I- tag at position {i}")
                    return False
                prev_type = labels[i-1][2:] if labels[i-1] != 'O' else None
                if prev_type != entity_type:
                    print(f"Warning: Invalid BIO transition at position {i}")
                    return False
        
        return True


class CoNLLPreprocessor:
    """
    Preprocessor for CoNLL-2003 dataset.
    Handles tokenization, alignment, and data preparation.
    """
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", 
                 max_length: int = 128):
        """
        Initialize preprocessor.
        
        Args:
            tokenizer_name: BERT tokenizer name
            max_length: Maximum sequence length
        """
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.bio_tagger = BIOTagger()
    
    def preprocess_conll_data(self, dataset) -> Dict:
        """
        Preprocess CoNLL-2003 dataset.
        
        Args:
            dataset: HuggingFace dataset object
            
        Returns:
            Dictionary with processed data
        """
        processed_data = {
            'tokens': [],
            'ner_tags': [],
            'bio_tags': []
        }
        
        for example in dataset:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Convert numerical NER tags to BIO format
            bio_tags = [self.bio_tagger.id_to_tag.get(tag, 'O') for tag in ner_tags]
            
            processed_data['tokens'].append(tokens)
            processed_data['ner_tags'].append(ner_tags)
            processed_data['bio_tags'].append(bio_tags)
        
        return processed_data
    
    def tokenize_with_alignment(self, tokens: List[str], labels: List[str]) -> Dict:
        """
        Tokenize with BERT tokenizer while maintaining label alignment.
        Handles subword tokenization.
        
        Args:
            tokens: Original word tokens
            labels: BIO labels for each token
            
        Returns:
            Dictionary with aligned tokens and labels
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get word IDs for alignment
        word_ids = encoding.word_ids()
        
        # Align labels with subwords
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP]) get label -100
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the original label
                aligned_labels.append(self.bio_tagger.tag_to_id.get(labels[word_idx], 0))
            else:
                # Subsequent subwords get -100 (ignore in loss)
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        encoding['labels'] = torch.tensor(aligned_labels)
        
        return encoding
    
    def prepare_data_for_crf(self, tokens: List[List[str]], labels: List[List[str]]):
        """
        Prepare features for CRF model.
        
        Args:
            tokens: List of token sequences
            labels: List of label sequences
        """
        def word2features(sent, i):
            """Extract features for a word at position i."""
            word = sent[i]
            
            features = {
                'bias': 1.0,
                'word.lower()': word.lower(),
                'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],
                'word.isupper()': word.isupper(),
                'word.istitle()': word.istitle(),
                'word.isdigit()': word.isdigit(),
            }
            
            if i > 0:
                prev_word = sent[i-1]
                features.update({
                    '-1:word.lower()': prev_word.lower(),
                    '-1:word.istitle()': prev_word.istitle(),
                    '-1:word.isupper()': prev_word.isupper(),
                })
            else:
                features['BOS'] = True
            
            if i < len(sent) - 1:
                next_word = sent[i+1]
                features.update({
                    '+1:word.lower()': next_word.lower(),
                    '+1:word.istitle()': next_word.istitle(),
                    '+1:word.isupper()': next_word.isupper(),
                })
            else:
                features['EOS'] = True
            
            return features
        
        def sent2features(sent):
            return [word2features(sent, i) for i in range(len(sent))]
        
        def sent2labels(sent_labels):
            return [label for label in sent_labels]
        
        X = [sent2features(s) for s in tokens]
        y = [sent2labels(l) for l in labels]
        
        return X, y