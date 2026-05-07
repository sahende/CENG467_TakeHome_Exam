"""
Qualitative analysis module for summarization (Q3).
Analyzes fluency, factual consistency, and information coverage.
"""
import numpy as np
from typing import List, Dict, Tuple
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


class SummaryAnalyzer:
    """
    Qualitative analysis of generated summaries.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.examples = []
    
    def add_example(self, source: str, reference: str, 
                    extractive_summary: str, abstractive_summary: str):
        """
        Add a comparison example.
        
        Args:
            source: Original source text
            reference: Reference summary
            extractive_summary: Extractive method summary
            abstractive_summary: Abstractive method summary
        """
        self.examples.append({
            'source': source,
            'reference': reference,
            'extractive': extractive_summary,
            'abstractive': abstractive_summary
        })
    
    def analyze_fluency(self, summary: str) -> Dict:
        """
        Analyze fluency aspects of a summary.
        
        Args:
            summary: Generated summary
            
        Returns:
            Dictionary with fluency metrics
        """
        sentences = summary.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        analysis = {
            'num_sentences': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'repeated_phrases': 0,
            'fragments': 0
        }
        
        # Check for repeated phrases
        words = summary.lower().split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if phrase in ' '.join(words[:i]) + ' ' + ' '.join(words[i+3:]):
                analysis['repeated_phrases'] += 1
        
        # Check for fragments (short sentences without verbs)
        for sentence in sentences:
            if len(sentence.split()) < 4:
                analysis['fragments'] += 1
        
        return analysis
    
    def analyze_information_coverage(self, source: str, summary: str) -> Dict:
        """
        Analyze how much of the source information is covered.
        
        Args:
            source: Source text
            summary: Generated summary
            
        Returns:
            Dictionary with coverage metrics
        """
        source_words = set(source.lower().split())
        summary_words = set(summary.lower().split())
        
        # Content word overlap (excluding stopwords)
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        
        source_content = source_words - stop_words
        summary_content = summary_words - stop_words
        
        overlap = source_content & summary_content
        
        return {
            'total_source_words': len(source_words),
            'total_summary_words': len(summary_words),
            'content_word_overlap': len(overlap),
            'coverage_ratio': len(overlap) / len(source_content) if source_content else 0
        }
    
    def generate_comparison_report(self) -> str:
        """
        Generate a detailed comparison report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("QUALITATIVE ANALYSIS: EXTRACTIVE vs ABSTRACTIVE SUMMARIZATION")
        report.append("=" * 80)
        
        for i, example in enumerate(self.examples, 1):
            report.append(f"\n{'='*60}")
            report.append(f"EXAMPLE {i}")
            report.append(f"{'='*60}")
            
            report.append(f"\n--- SOURCE TEXT (first 200 chars) ---")
            report.append(example['source'][:200] + "...")
            
            report.append(f"\n--- REFERENCE SUMMARY ---")
            report.append(example['reference'])
            
            report.append(f"\n--- EXTRACTIVE SUMMARY ---")
            report.append(example['extractive'])
            
            report.append(f"\n--- ABSTRACTIVE SUMMARY ---")
            report.append(example['abstractive'])
            
            # Analyze fluency
            report.append("\n--- FLUENCY ANALYSIS ---")
            ext_fluency = self.analyze_fluency(example['extractive'])
            abs_fluency = self.analyze_fluency(example['abstractive'])
            
            report.append(f"Extractive - Sentences: {ext_fluency['num_sentences']}, "
                         f"Avg Length: {ext_fluency['avg_sentence_length']:.1f}")
            report.append(f"Abstractive - Sentences: {abs_fluency['num_sentences']}, "
                         f"Avg Length: {abs_fluency['avg_sentence_length']:.1f}")
            
            # Analyze coverage
            report.append("\n--- INFORMATION COVERAGE ---")
            ext_coverage = self.analyze_information_coverage(
                example['source'], example['extractive']
            )
            abs_coverage = self.analyze_information_coverage(
                example['source'], example['abstractive']
            )
            
            report.append(f"Extractive - Coverage Ratio: {ext_coverage['coverage_ratio']:.3f}")
            report.append(f"Abstractive - Coverage Ratio: {abs_coverage['coverage_ratio']:.3f}")
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save analysis report."""
        report = self.generate_comparison_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Analysis report saved to {filepath}")


if __name__ == "__main__":
    """
    Run qualitative analysis on saved Q3 results.
    """
    import json
    
    results_path = os.path.join(RESULTS_DIR, 'q3_results.json')
    
    if os.path.exists(results_path):
        print("Loading Q3 results for analysis...")
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        analyzer = SummaryAnalyzer()
        
        # Bu kısım için train.py'den gelen örnekler lazım
        # Onları da q3_qualitative_examples.json olarak kaydedelim
        examples_path = os.path.join(RESULTS_DIR, 'q3_qualitative_examples.json')
        if os.path.exists(examples_path):
            with open(examples_path, 'r') as f:
                examples = json.load(f)
            
            for ex in examples:
                analyzer.add_example(
                    source=ex['source'],
                    reference=ex['reference'],
                    extractive_summary=ex['extractive'],
                    abstractive_summary=ex['abstractive']
                )
            
            print(analyzer.generate_comparison_report())
            analyzer.save_report(os.path.join(RESULTS_DIR, 'q3_qualitative_analysis.txt'))
        else:
            print("No qualitative examples found. Run train.py first.")
    else:
        print("No Q3 results found. Run train.py first.")