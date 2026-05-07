"""
Error analysis module for NER (Q2).
Analyzes boundary errors and entity confusion.
"""
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from seqeval.metrics import classification_report
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


class NERErrorAnalyzer:
    """
    Analyzes NER errors including boundary errors and entity confusion.
    """
    
    def __init__(self, true_labels: List[List[str]], 
                 predicted_labels: List[List[str]]):
        """
        Initialize error analyzer.
        
        Args:
            true_labels: List of true label sequences
            predicted_labels: List of predicted label sequences
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        
        # Extract entities from BIO tags
        self.true_entities = self._extract_entities(true_labels)
        self.pred_entities = self._extract_entities(predicted_labels)
    
    def _extract_entities(self, labels: List[List[str]]) -> List[Set]:
        """
        Extract entities from BIO label sequences.
        
        Returns:
            List of sets, each containing (entity_type, start, end) tuples
        """
        entities = []
        
        for seq in labels:
            seq_entities = set()
            i = 0
            while i < len(seq):
                if seq[i].startswith('B-'):
                    entity_type = seq[i][2:]  # PER, ORG, LOC, MISC
                    start = i
                    i += 1
                    # Collect I- tags of same type
                    while i < len(seq) and seq[i] == f'I-{entity_type}':
                        i += 1
                    seq_entities.add((entity_type, start, i))
                else:
                    i += 1
            entities.append(seq_entities)
        
        return entities
    
    def analyze_boundary_errors(self) -> Dict:
        """
        Analyze boundary errors in NER predictions.
        
        Boundary error types:
        - Exact match: Same type, same span
        - Right type, wrong boundary: Same type, overlapping span
        - Partial match: Different type, overlapping span
        - Missed: Entity in true but not predicted
        - Extra: Entity predicted but not in true
        
        Returns:
            Dictionary with boundary error statistics
        """
        stats = {
            'exact_matches': 0,
            'right_type_wrong_boundary': 0,
            'partial_matches': 0,  # Wrong type but overlapping span
            'missed_entities': 0,
            'extra_entities': 0,
            'total_true_entities': 0,
            'total_pred_entities': 0,
            'boundary_error_examples': []
        }
        
        for true_set, pred_set in zip(self.true_entities, self.pred_entities):
            stats['total_true_entities'] += len(true_set)
            stats['total_pred_entities'] += len(pred_set)
            
            matched_pred = set()
            matched_true = set()
            
            for true_entity in true_set:
                true_type, true_start, true_end = true_entity
                found = False
                
                for pred_entity in pred_set:
                    if pred_entity in matched_pred:
                        continue
                    
                    pred_type, pred_start, pred_end = pred_entity
                    
                    # Check overlap
                    overlap = (true_start < pred_end and pred_start < true_end)
                    
                    if true_type == pred_type:
                        if true_start == pred_start and true_end == pred_end:
                            # Exact match
                            stats['exact_matches'] += 1
                            matched_pred.add(pred_entity)
                            matched_true.add(true_entity)
                            found = True
                        elif overlap:
                            # Right type, wrong boundary
                            stats['right_type_wrong_boundary'] += 1
                            matched_pred.add(pred_entity)
                            matched_true.add(true_entity)
                            found = True
                            # Save example
                            if len(stats['boundary_error_examples']) < 5:
                                stats['boundary_error_examples'].append({
                                    'true': f"{true_type}[{true_start}:{true_end}]",
                                    'pred': f"{pred_type}[{pred_start}:{pred_end}]",
                                    'error': 'boundary'
                                })
                    elif overlap:
                        # Different type, overlapping
                        stats['partial_matches'] += 1
                        matched_pred.add(pred_entity)
                        matched_true.add(true_entity)
            
            # Count missed/extra
            stats['missed_entities'] += len(true_set - matched_true)
            stats['extra_entities'] += len(pred_set - matched_pred)
        
        # Calculate rates
        total = stats['total_true_entities']
        if total > 0:
            stats['exact_match_rate'] = stats['exact_matches'] / total
            stats['boundary_error_rate'] = stats['right_type_wrong_boundary'] / total
            stats['miss_rate'] = stats['missed_entities'] / total
        
        return stats
    
    def analyze_entity_confusion(self) -> Dict:
        """
        Analyze entity type confusion.
        
        Returns:
            Confusion matrix dictionary
        """
        confusion = defaultdict(lambda: defaultdict(int))
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        
        # Only look at exact span matches
        for true_set, pred_set in zip(self.true_entities, self.pred_entities):
            # Create span -> type mappings
            true_span_map = {(s, e): t for t, s, e in true_set}
            pred_span_map = {(s, e): t for t, s, e in pred_set}
            
            # Find matching spans
            common_spans = set(true_span_map.keys()) & set(pred_span_map.keys())
            
            for span in common_spans:
                true_type = true_span_map[span]
                pred_type = pred_span_map[span]
                confusion[true_type][pred_type] += 1
        
        # Convert to regular dict
        confusion_dict = {}
        for t1 in entity_types:
            confusion_dict[t1] = {}
            for t2 in entity_types:
                confusion_dict[t1][t2] = confusion[t1][t2]
        
        return confusion_dict
    
    
    def generate_report(self) -> str:
        """Generate detailed error analysis report."""
        report = []
        report.append("=" * 60)
        report.append("NER ERROR ANALYSIS REPORT")
        report.append("=" * 60)
        
        # 1. Boundary errors
        boundary = self.analyze_boundary_errors()
        report.append(f"\n{'─'*40}")
        report.append("1. BOUNDARY ERROR ANALYSIS")
        report.append(f"{'─'*40}")
        report.append(f"  Total true entities:    {boundary['total_true_entities']}")
        report.append(f"  Total predicted entities: {boundary['total_pred_entities']}")
        report.append(f"")
        report.append(f"   Exact matches:          {boundary['exact_matches']} ({boundary.get('exact_match_rate', 0):.1%})")
        report.append(f"   Right type, wrong span: {boundary['right_type_wrong_boundary']} ({boundary.get('boundary_error_rate', 0):.1%})")
        report.append(f"   Partial matches (wrong type): {boundary['partial_matches']}")
        report.append(f"   Missed entities:        {boundary['missed_entities']} ({boundary.get('miss_rate', 0):.1%})")
        report.append(f"   Extra (hallucinated):   {boundary['extra_entities']}")
        
        if boundary['boundary_error_examples']:
            report.append(f"\n  Boundary Error Examples:")
            for i, ex in enumerate(boundary['boundary_error_examples'][:3], 1):
                report.append(f"    {i}. True: {ex['true']} → Pred: {ex['pred']}")
        
        # 2. Entity confusion
        confusion = self.analyze_entity_confusion()
        report.append(f"\n{'─'*40}")
        report.append("2. ENTITY TYPE CONFUSION MATRIX")
        report.append(f"{'─'*40}")
        
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        
        # Header
        header = f"  {'True\\Pred':>10}"
        for t in entity_types:
            header += f" {t:>8}"
        report.append(header)
        report.append(f"  {'─'*42}")
        
        # Rows
        for t1 in entity_types:
            row = f"  {t1:>10}"
            for t2 in entity_types:
                count = confusion[t1][t2]
                marker = "✓" if t1 == t2 else "✗"
                row += f" {count:>4} {marker} "
            report.append(row)
        
        # 3. Per-entity metrics
        report.append(f"\n{'─'*40}")
        report.append("3. PER-ENTITY PERFORMANCE")
        report.append(f"{'─'*40}")
        report.append(classification_report(self.true_labels, self.predicted_labels))
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save error analysis report to file."""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {filepath}")

def plot_entity_confusion_matrix(confusion_dict: Dict, model_name: str, save_path: str):
    """Plot entity type confusion matrix as heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    entity_types = ['PER', 'ORG', 'LOC', 'MISC']
    matrix = np.zeros((4, 4))
    
    for i, t1 in enumerate(entity_types):
        for j, t2 in enumerate(entity_types):
            matrix[i][j] = confusion_dict[t1][t2]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=entity_types, yticklabels=entity_types,
                cbar_kws={'label': 'Count'})
    plt.title(f'Entity Type Confusion Matrix – {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Entity Type')
    plt.xlabel('Predicted Entity Type')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix plot saved to {save_path}")


def plot_boundary_comparison(all_analyses: Dict, save_path: str):
    """Plot boundary error comparison across models."""
    import matplotlib.pyplot as plt
    
    models = list(all_analyses.keys())
    metrics = ['exact_matches', 'right_type_wrong_boundary', 'missed_entities', 'extra_entities']
    labels = ['Exact Matches', 'Wrong Boundary', 'Missed', 'Extra']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [all_analyses[m]['boundary'][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val),
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Count')
    ax.set_title('Boundary Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Boundary comparison plot saved to {save_path}")




def main():
    """
    Main function - loads predictions from train.py results and runs error analysis.
    """
    from datasets import load_dataset
    from q2_ner.preprocess import BIOTagger
    import json
    
    print("=" * 60)
    print("Q2 NER ERROR ANALYSIS")
    print("=" * 60)
    
    # Load predictions saved by train.py
    preds_file = os.path.join(RESULTS_DIR, 'q2_predictions.json')
    
    if not os.path.exists(preds_file):
        print(f" No predictions found at {preds_file}")
        print("   Run train.py first to generate predictions.")
        return
    
    print(f"Loading saved predictions from {preds_file}")
    with open(preds_file, 'r') as f:
        saved_preds = json.load(f)
    
    print(f"Found {len(saved_preds)} models to analyze: {list(saved_preds.keys())}\n")
    
    all_analyses = {}
    
    for model_name, data in saved_preds.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*60}")
        
        true_labels = data['true_labels']
        predictions = data['predictions']
        
        # Quick stats
        from seqeval.metrics import precision_score, recall_score, f1_score
        p = precision_score(true_labels, predictions)
        r = recall_score(true_labels, predictions)
        f = f1_score(true_labels, predictions)
        
        print(f"\n  Overall Metrics:")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall:    {r:.4f}")
        print(f"    F1-Score:  {f:.4f}")
        
        # Create analyzer
        analyzer = NERErrorAnalyzer(true_labels, predictions)
        
        # Print full report
        print(analyzer.generate_report())
        
        # Save individual report
        report_path = os.path.join(RESULTS_DIR, f'q2_error_analysis_{model_name}.txt')
        analyzer.save_report(report_path)
        
        # Save boundary analysis for comparison
        all_analyses[model_name] = {
            'metrics': {'precision': p, 'recall': r, 'f1': f},
            'boundary': analyzer.analyze_boundary_errors(),
            'confusion': analyzer.analyze_entity_confusion()
        }
    
    # Print cross-model comparison
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'─'*50}")
    for model_name, analysis in all_analyses.items():
        m = analysis['metrics']
        print(f"{model_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    # Compare boundary errors
    print(f"\n{'─'*50}")
    print(f"BOUNDARY ERROR COMPARISON:")
    print(f"{'Model':<15} {'Exact':<8} {'Boundary':<10} {'Missed':<8} {'Extra':<8}")
    print(f"{'─'*50}")
    for model_name, analysis in all_analyses.items():
        b = analysis['boundary']
        print(f"{model_name:<15} {b['exact_matches']:<8} {b['right_type_wrong_boundary']:<10} "
              f"{b['missed_entities']:<8} {b['extra_entities']:<8}")
    
    # Common errors between models
    if len(saved_preds) >= 2:
        model_names = list(saved_preds.keys())
        print(f"\n{'─'*50}")
        print(f"COMMON ERROR ANALYSIS ({model_names[0]} vs {model_names[1]}):")
        
        preds_1 = saved_preds[model_names[0]]['predictions']
        preds_2 = saved_preds[model_names[1]]['predictions']
        true = saved_preds[model_names[0]]['true_labels']
        
        both_wrong = 0
        for t, p1, p2 in zip(true, preds_1, preds_2):
            if p1 != t and p2 != t:
                both_wrong += 1
        
        total_wrong_1 = sum(1 for t, p in zip(true, preds_1) if p != t)
        total_wrong_2 = sum(1 for t, p in zip(true, preds_2) if p != t)
        
        print(f"  {model_names[0]} errors: {total_wrong_1} sentences")
        print(f"  {model_names[1]} errors: {total_wrong_2} sentences")
        print(f"  Both wrong on same sentences: {both_wrong}")
        if total_wrong_1 > 0:
            print(f"  Overlap: {both_wrong/total_wrong_1*100:.1f}% of {model_names[0]}'s errors")
    
    # Save combined analysis
    combined_path = os.path.join(RESULTS_DIR, 'q2_error_analysis_combined.json')
    
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, set): return list(obj)
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        return obj
    
    with open(combined_path, 'w') as f:
        json.dump(convert(all_analyses), f, indent=4)
    print(f"\n Combined analysis saved to {combined_path}")

    # Generate plots
    for model_name, analysis in all_analyses.items():
        cm_path = os.path.join(RESULTS_DIR, f'q2_confusion_matrix_{model_name}.png')
        plot_entity_confusion_matrix(analysis['confusion'], model_name, cm_path)
    
    boundary_path = os.path.join(RESULTS_DIR, 'q2_boundary_comparison.png')
    plot_boundary_comparison(all_analyses, boundary_path)
    print(f"\n{'='*60}")
    print("Q2 Error Analysis Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()