"""
Evaluation module for summarization (Q3).
Computes multiple metrics: ROUGE, BLEU, METEOR, BERTScore.
"""
import transformers

if not hasattr(transformers.BertTokenizer, 'build_inputs_with_special_tokens'):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Re-add removed method for bert-score compatibility."""
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
    
    transformers.BertTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
    transformers.BertTokenizerFast.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
    print("✓ Patched BertTokenizer for bert-score compatibility")
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score as bert_score
from typing import List, Dict, Union
import numpy as np

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class SummarizationEvaluator:
    """
    Comprehensive summarization evaluator with multiple metrics.
    """
    
    def __init__(self):
        """Initialize all scorers."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smooth_fn = SmoothingFunction()
    
    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bleu(self, reference: str, hypothesis: str, 
                     weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
        """
        Compute BLEU score with smoothing.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            weights: N-gram weights
            
        Returns:
            BLEU score
        """
        reference_tokens = [reference.split()]
        hypothesis_tokens = hypothesis.split()
        
        try:
            bleu = sentence_bleu(
                reference_tokens,
                hypothesis_tokens,
                weights=weights,
                smoothing_function=self.smooth_fn.method1
            )
        except:
            bleu = 0.0
        
        return bleu
    
    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Compute METEOR score using nltk.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            METEOR score
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            reference_tokens = reference.split()
            hypothesis_tokens = hypothesis.split()
            
            if len(hypothesis_tokens) == 0:
                return 0.0
            
            meteor = meteor_score([reference_tokens], hypothesis_tokens)
            return meteor
        except:
            return 0.0
    
    def compute_bertscore(self, references: List[str], 
                      hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            
        Returns:
            Dictionary with BERTScore metrics
        """
        try:
            # CRITICAL: Convert to plain Python lists (fix Column type error)
            refs_clean = [str(r) for r in references]
            hyps_clean = [str(h) for h in hypotheses]
            
            P, R, F1 = bert_score(
                hyps_clean, 
                refs_clean, 
                lang='en',
                model_type='bert-base-uncased',
                verbose=False
            )
            
            return {
                'bert_score_precision': P.mean().item(),
                'bert_score_recall': R.mean().item(),
                'bert_score_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return {
                'bert_score_precision': 0.0,
                'bert_score_recall': 0.0,
                'bert_score_f1': 0.0
            }
    
    def evaluate_all(self, references: List[str], 
                     hypotheses: List[str]) -> Dict[str, Dict]:
        """
        Run all evaluation metrics.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            
        Returns:
            Dictionary with all metrics
        """
        all_metrics = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': [],
            'meteor': []
        }
        
        # Compute per-sample metrics
        for ref, hyp in zip(references, hypotheses):
            # ROUGE
            rouge = self.compute_rouge(ref, hyp)
            all_metrics['rouge1'].append(rouge['rouge1'])
            all_metrics['rouge2'].append(rouge['rouge2'])
            all_metrics['rougeL'].append(rouge['rougeL'])
            
            # BLEU
            bleu = self.compute_bleu(ref, hyp)
            all_metrics['bleu'].append(bleu)
            
            # METEOR
            meteor = self.compute_meteor(ref, hyp)
            all_metrics['meteor'].append(meteor)
        
        # Compute BERTScore
        bert_scores = self.compute_bertscore(references, hypotheses)
        all_metrics.update({
            'bert_score_precision': bert_scores['bert_score_precision'],
            'bert_score_recall': bert_scores['bert_score_recall'],
            'bert_score_f1': bert_scores['bert_score_f1']
        })
        
        # Calculate averages
        summary = {}
        for metric, scores in all_metrics.items():
            if isinstance(scores, list):
                summary[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            else:
                summary[metric] = scores
        
        return summary