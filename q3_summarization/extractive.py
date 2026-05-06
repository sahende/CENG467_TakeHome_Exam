"""
Extractive summarization module (Q3).
Implements TextRank algorithm.
"""
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple

nltk.download('punkt', quiet=True)


class TextRankSummarizer:
    """
    TextRank-based extractive summarization.
    """
    
    def __init__(self, similarity_threshold: float = 0.1, 
                 damping_factor: float = 0.85,
                 max_iterations: int = 100):
        """
        Initialize TextRank summarizer.
        
        Args:
            similarity_threshold: Minimum similarity for edge creation
            damping_factor: Damping factor for PageRank
            max_iterations: Maximum iterations for convergence
        """
        self.similarity_threshold = similarity_threshold
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate cosine similarity between two sentences using TF-IDF.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Cosine similarity score
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform([sent1, sent2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0]
        except:
            return 0.0
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build sentence similarity matrix.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim = self._sentence_similarity(sentences[i], sentences[j])
                    if sim > self.similarity_threshold:
                        similarity_matrix[i][j] = sim
        
        return similarity_matrix
    
    def _pagerank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute PageRank scores from similarity matrix.
        
        Args:
            similarity_matrix: Sentence similarity matrix
            
        Returns:
            Array of PageRank scores
        """
        # Normalize similarity matrix
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / row_sums
        
        n = len(similarity_matrix)
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Power iteration
        for _ in range(self.max_iterations):
            prev_scores = scores.copy()
            scores = (1 - self.damping_factor) / n + \
                     self.damping_factor * transition_matrix.T @ scores
            
            # Check convergence
            if np.abs(scores - prev_scores).sum() < 1e-6:
                break
        
        return scores
    
    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        """
        Generate extractive summary using TextRank.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            List of selected sentences in original order
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Build similarity matrix and compute scores
        similarity_matrix = self._build_similarity_matrix(sentences)
        scores = self._pagerank(similarity_matrix)
        
        # Select top sentences
        ranked_indices = np.argsort(scores)[::-1]
        selected_indices = sorted(ranked_indices[:num_sentences])
        
        # Return selected sentences in original order
        summary = [sentences[i] for i in selected_indices]
        
        return summary


