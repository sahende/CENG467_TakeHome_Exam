"""
Preprocessing module for text classification (Q1).
Implements multiple tokenization strategies and preprocessing pipelines.
"""
import re
import string
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

# These downloads ensure the code runs without manual intervention.
# Required for: stopword removal, tokenization, METEOR scoring

NLTK_RESOURCES = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']

for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                       f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4'] else
                       f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        nltk.download(resource, quiet=True)
        print(f"  ✓ {resource} downloaded successfully")

print("NLTK resources ready.")


class TextPreprocessor:
    """
    Preprocessing pipeline for text classification with multiple strategies.
    """
    
    def __init__(self, tokenizer_type: str = "basic", max_length: int = 512):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer_type: "basic", "nltk", "bert_tokenizer"
            max_length: Maximum sequence length
        """
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize BERT tokenizer if needed
        if tokenizer_type == "bert_tokenizer":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def normalize_text(self, text: str, lowercase: bool = True, 
                       remove_punctuation: bool = True) -> str:
        """
        Normalize text by applying various transformations.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            
        Returns:
            Normalized text
        """
        if lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize_basic(self, text: str) -> List[str]:
        """Basic whitespace tokenization."""
        return text.split()
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """NLTK-based tokenization."""
        return word_tokenize(text)
    
    def tokenize_bert(self, text: str) -> Dict:
        """
        BERT tokenization with special tokens and attention masks.
        
        Returns:
            Dictionary with input_ids, attention_mask
        """
        tokens = self.bert_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        return tokens
    
    def remove_stopwords(self, tokens: List[str], remove: bool = True) -> List[str]:
        """Remove stopwords from token list."""
        if remove:
            return [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def preprocess(self, text: str, remove_stop: bool = False) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            remove_stop: Whether to remove stopwords
            
        Returns:
            Dictionary with processed tokens based on tokenizer type
        """
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Tokenize based on strategy
        if self.tokenizer_type == "bert_tokenizer":
            tokens = self.tokenize_bert(normalized)
            tokens['original_text'] = text
            return tokens
        elif self.tokenizer_type == "nltk":
            tokens = self.tokenize_nltk(normalized)
        else:
            tokens = self.tokenize_basic(normalized)
        
        # Apply stopword removal
        if remove_stop:
            tokens = self.remove_stopwords(tokens)
        
        # Truncate if necessary
        tokens = tokens[:self.max_length]
        
        return {
            'tokens': tokens,
            'text': ' '.join(tokens),
            'original_text': text
        }


class TFIDFVectorizer:
    """TF-IDF vectorization wrapper."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform text data."""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new text data."""
        return self.vectorizer.transform(texts)


def prepare_dataset_for_training(texts: List[str], labels: List[int], 
                                 preprocessor: TextPreprocessor,
                                 tokenizer_type: str = "basic") -> Tuple:
    """
    Prepare dataset for model training.
    
    Args:
        texts: List of text samples
        labels: List of labels
        preprocessor: TextPreprocessor instance
        tokenizer_type: Type of tokenizer to use
        
    Returns:
        Tuple of processed data and labels
    """
    processed_texts = []
    processed_labels = []
    
    for text, label in zip(texts, labels):
        result = preprocessor.preprocess(text)
        processed_texts.append(result)
        processed_labels.append(label)
    
    return processed_texts, processed_labels