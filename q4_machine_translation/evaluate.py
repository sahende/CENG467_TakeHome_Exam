"""
Evaluation module for machine translation (Q4).
Computes BLEU, METEOR, ChrF, and BERTScore.
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
from typing import List, Dict
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import nltk
nltk.download('wordnet', quiet=True)


class TranslationEvaluator:
    """
    Comprehensive MT evaluation suite.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def compute_bleu(self, references: List[str], 
                     hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score using sacreBLEU.
        
        Args:
            references: List of reference translations
            hypotheses: List of candidate translations
            
        Returns:
            Dictionary with BLEU scores
        """
        # sacreBLEU expects references as list of lists
        if isinstance(references[0], str):
            references = [[r] for r in references]
        
        bleu = corpus_bleu(hypotheses, references)
        
        return {
            'bleu': bleu.score,
            'bleu_signature': str(bleu),
            'bleu_precisions': list(bleu.precisions),
            'bleu_bp': bleu.bp
        }
    
    def compute_meteor(self, references: List[str], 
                       hypotheses: List[str]) -> float:
        """
        Compute METEOR score.
        
        Args:
            references: Reference translations
            hypotheses: Candidate translations
            
        Returns:
            Average METEOR score
        """
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            
            if len(hyp_tokens) == 0:
                scores.append(0.0)
            else:
                score = meteor_score([ref_tokens], hyp_tokens)
                scores.append(score)
        
        return np.mean(scores)
    
    def compute_chrf(self, references: List[str], 
                     hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ChrF score (character n-gram F-score).
        
        Args:
            references: Reference translations
            hypotheses: Candidate translations
            
        Returns:
            Dictionary with ChrF scores
        """
        if isinstance(references[0], str):
            references = [[r] for r in references]
        
        chrf = corpus_chrf(hypotheses, references)
        
        return {
            'chrf': chrf.score,
            'chrf_char_order': 6,
            'chrf_word_order': 2,
            'chrf_beta': 2
        }
    
    def compute_bertscore(self, references: List[str], 
                      hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.
        Uses BERT model to compare reference and hypothesis embeddings.
        """
        try:
            # Clean inputs
            refs_clean = [str(r) for r in references]
            hyps_clean = [str(h) for h in hypotheses]
            
            # BERTScore
            P, R, F1 = bert_score(
                hyps_clean, 
                refs_clean, 
                lang='de',
                model_type='bert-base-multilingual-cased',  
                verbose=False,
                device='cpu' if not torch.cuda.is_available() else 'cuda'
            )
            
            return {
                'bert_score_f1': F1.mean().item(),
                'bert_score_precision': P.mean().item(),
                'bert_score_recall': R.mean().item()
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            import traceback
            traceback.print_exc()
            f1_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = set(str(ref).lower().split())
                hyp_tokens = set(str(hyp).lower().split())
                if ref_tokens and hyp_tokens:
                    overlap = len(ref_tokens & hyp_tokens)
                    prec = overlap / len(hyp_tokens) if hyp_tokens else 0
                    rec = overlap / len(ref_tokens) if ref_tokens else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    f1_scores.append(f1)
                else:
                    f1_scores.append(0.0)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            print(f"  Fallback token overlap F1: {avg_f1:.4f}")
            return {
                'bert_score_f1': avg_f1,
                'bert_score_precision': avg_f1,
                'bert_score_recall': avg_f1
            }
    
    def evaluate_all(self, references: List[str], 
                     hypotheses: List[str]) -> Dict:
        """
        Compute all metrics.
        
        Args:
            references: Reference translations
            hypotheses: Candidate translations
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # BLEU
        bleu_results = self.compute_bleu(references, hypotheses)
        results.update(bleu_results)
        
        # METEOR
        results['meteor'] = self.compute_meteor(references, hypotheses)
        
        # ChrF
        chrf_results = self.compute_chrf(references, hypotheses)
        results.update(chrf_results)
        
        # BERTScore
        bert_results = self.compute_bertscore(references, hypotheses)
        results.update(bert_results)
        
        return results
    
    def compare_models(self, references: List[str],
                       seq2seq_hypotheses: List[str],
                       transformer_hypotheses: List[str]) -> Dict:
        """
        Compare two MT models.
        
        Args:
            references: Reference translations
            seq2seq_hypotheses: Seq2Seq model outputs
            transformer_hypotheses: Transformer model outputs
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*60)
        print("MT MODEL COMPARISON")
        print("="*60)
        
        comparison = {}
        
        # Evaluate Seq2Seq
        print("\n--- Seq2Seq with Attention ---")
        seq2seq_metrics = self.evaluate_all(references, seq2seq_hypotheses)
        comparison['seq2seq'] = seq2seq_metrics
        
        # Evaluate Transformer
        print("\n--- Transformer-based Model ---")
        transformer_metrics = self.evaluate_all(references, transformer_hypotheses)
        comparison['transformer'] = transformer_metrics
        
        # Print comparison
        print("\n--- Metrics Summary ---")
        metrics_to_compare = ['bleu', 'meteor', 'chrf', 'bert_score_f1']
        
        for metric in metrics_to_compare:
            if metric in comparison['seq2seq']:
                seq2seq_val = comparison['seq2seq'][metric]
                trans_val = comparison['transformer'].get(metric, float('nan'))
                print(f"{metric:15} | Seq2Seq: {seq2seq_val:.4f} | "
                      f"Transformer: {trans_val:.4f}")
        
        return comparison