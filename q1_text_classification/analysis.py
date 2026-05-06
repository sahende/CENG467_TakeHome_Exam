"""
Error analysis module for text classification (Q1).
Analyzes misclassified examples and common error patterns across ALL models.
Includes preprocessing impact analysis, model comparison, and cross-model error analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class ErrorAnalyzer:
    """
    Analyzes misclassified examples and identifies error patterns.
    Works with any model output.
    """
    
    def __init__(self, texts: List[str], true_labels: List[int], 
                 predicted_labels: List[int], 
                 predictions_proba: Optional[np.ndarray] = None,
                 model_name: str = "Unknown"):
        """Initialize error analyzer."""
        self.texts = texts
        self.true_labels = np.array(true_labels)
        self.predicted_labels = np.array(predicted_labels)
        self.predictions_proba = predictions_proba
        self.model_name = model_name
        
        self.error_mask = self.true_labels != self.predicted_labels
        self.misclassified_indices = np.where(self.error_mask)[0]
        self.correct_indices = np.where(~self.error_mask)[0]
        
        self.df = pd.DataFrame({
            'index': range(len(texts)),
            'text': texts,
            'text_length': [len(t.split()) for t in texts],
            'true_label': true_labels,
            'predicted_label': predicted_labels,
            'is_misclassified': self.error_mask
        })
        
        if predictions_proba is not None:
            self.df['confidence'] = np.max(predictions_proba, axis=1)
    
    def get_misclassified_examples(self, n_examples: int = 5, 
                                   sort_by: str = 'confidence') -> pd.DataFrame:
        """Get n misclassified examples with details."""
        misclassified_df = self.df[self.df['is_misclassified']].copy()
        
        if len(misclassified_df) == 0:
            print(f"  No misclassified examples found!")
            return pd.DataFrame()
        
        if sort_by == 'confidence' and 'confidence' in misclassified_df.columns:
            misclassified_df = misclassified_df.sort_values('confidence', ascending=False)
        elif sort_by == 'length':
            misclassified_df = misclassified_df.sort_values('text_length', ascending=False)
        
        result = misclassified_df.head(n_examples)
        available_columns = ['text', 'text_length', 'true_label', 'predicted_label']
        if 'confidence' in result.columns:
            available_columns.append('confidence')
        
        return result[available_columns]
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze common error patterns systematically."""
        analysis = {
            'model_name': self.model_name,
            'total_samples': len(self.true_labels),
            'total_errors': int(np.sum(self.error_mask)),
            'error_rate': float(np.mean(self.error_mask)),
            'total_correct': int(np.sum(~self.error_mask)),
            'accuracy': float(np.mean(~self.error_mask))
        }
        
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        analysis['confusion_matrix'] = cm.tolist()
        
        if len(np.unique(self.true_labels)) == 2:
            fp_mask = (self.true_labels == 0) & (self.predicted_labels == 1)
            fn_mask = (self.true_labels == 1) & (self.predicted_labels == 0)
            
            analysis['false_positives'] = int(np.sum(fp_mask))
            analysis['false_negatives'] = int(np.sum(fn_mask))
            
            total_pos = int(np.sum(self.true_labels == 1))
            total_neg = int(np.sum(self.true_labels == 0))
            
            analysis['fp_rate'] = float(np.sum(fp_mask) / total_neg) if total_neg > 0 else 0
            analysis['fn_rate'] = float(np.sum(fn_mask) / total_pos) if total_pos > 0 else 0
        
        text_lengths = self.df['text_length'].values
        error_text_lengths = text_lengths[self.error_mask]
        correct_text_lengths = text_lengths[~self.error_mask]
        
        analysis['text_length_analysis'] = {
            'avg_length_all': float(np.mean(text_lengths)),
            'avg_length_errors': float(np.mean(error_text_lengths)) if len(error_text_lengths) > 0 else 0,
            'avg_length_correct': float(np.mean(correct_text_lengths)) if len(correct_text_lengths) > 0 else 0,
        }
        
        if self.predictions_proba is not None:
            error_confidence = self.df[self.df['is_misclassified']]['confidence'].values
            correct_confidence = self.df[~self.df['is_misclassified']]['confidence'].values
            
            analysis['confidence_analysis'] = {
                'avg_confidence_errors': float(np.mean(error_confidence)) if len(error_confidence) > 0 else 0,
                'avg_confidence_correct': float(np.mean(correct_confidence)) if len(correct_confidence) > 0 else 0,
            }
        
        return analysis
    
    def identify_error_categories(self) -> Dict[str, List[int]]:
        """Categorize misclassified examples by error patterns."""
        categories = {
            'short_text_errors': [],
            'long_text_errors': [],
            'false_positives': [],
            'false_negatives': []
        }
        
        misclassified = self.df[self.df['is_misclassified']]
        
        for idx, row in misclassified.iterrows():
            if row['text_length'] < 30:
                categories['short_text_errors'].append(idx)
            elif row['text_length'] > 250:
                categories['long_text_errors'].append(idx)
            
            if row['true_label'] == 0 and row['predicted_label'] == 1:
                categories['false_positives'].append(idx)
            elif row['true_label'] == 1 and row['predicted_label'] == 0:
                categories['false_negatives'].append(idx)
        
        return categories
    
    def analyze_error_content_patterns(self) -> Dict:
        """Analyze content patterns in misclassified examples."""
        misclassified = self.df[self.df['is_misclassified']]
        
        patterns = {
            'negation_heavy': [],
            'mixed_sentiment': [],
            'very_short': [],
            'very_long': []
        }
        
        negation_words = ['not', "n't", 'never', 'no', 'neither', 'hardly']
        contrast_words = ['but', 'however', 'although', 'though', 'while', 'despite']
        
        for idx, row in misclassified.iterrows():
            text = str(row['text']).lower()
            
            negation_count = sum(1 for w in negation_words if f' {w} ' in f' {text} ')
            if negation_count >= 2:
                patterns['negation_heavy'].append(idx)
            
            contrast_count = sum(1 for w in contrast_words if f' {w} ' in f' {text} ')
            if contrast_count >= 1:
                patterns['mixed_sentiment'].append(idx)
            
            if row['text_length'] < 30:
                patterns['very_short'].append(idx)
            elif row['text_length'] > 300:
                patterns['very_long'].append(idx)
        
        return patterns
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                             class_names: List[str] = None):
        """Plot confusion matrix."""
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'
        
        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # plt.show() yerine kapat
    
    def plot_error_distribution(self, save_path: Optional[str] = None):
        """Plot error type distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.df[self.df['is_misclassified']]['text_length'], 
                    bins=20, alpha=0.7, label='Misclassified', color='red', edgecolor='black')
        axes[0].hist(self.df[~self.df['is_misclassified']]['text_length'], 
                    bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0].set_xlabel('Text Length (words)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Text Length Distribution - {self.model_name}')
        axes[0].legend()
        
        if len(np.unique(self.true_labels)) == 2:
            fp = np.sum((self.true_labels == 0) & (self.predicted_labels == 1))
            fn = np.sum((self.true_labels == 1) & (self.predicted_labels == 0))
            
            axes[1].pie([fp, fn], labels=['False Positives', 'False Negatives'],
                       autopct='%1.1f%%', colors=['#ff6b6b', '#ffa502'],
                       startangle=90, explode=(0.05, 0.05))
            axes[1].set_title(f'Error Type Distribution - {self.model_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_misclassified(self, save_path: str, n_examples: int = 10):
        """Export misclassified examples to CSV."""
        misclassified = self.get_misclassified_examples(n_examples)
        if len(misclassified) > 0:
            misclassified.to_csv(save_path, index=False)
            print(f"  Misclassified examples saved to {save_path}")
    
    def print_report(self):
        """Print comprehensive error analysis report."""
        analysis = self.analyze_error_patterns()
        categories = self.identify_error_categories()
        content_patterns = self.analyze_error_content_patterns()
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS REPORT: {self.model_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Samples: {analysis['total_samples']}")
        print(f"  Correct: {analysis['total_correct']} ({analysis['accuracy']:.2%})")
        print(f"  Errors: {analysis['total_errors']} ({analysis['error_rate']:.2%})")
        
        if 'false_positives' in analysis:
            print(f"\n📋 ERROR BREAKDOWN:")
            print(f"  False Positives (0→1): {analysis['false_positives']}")
            print(f"  False Negatives (1→0): {analysis['false_negatives']}")
        
        tla = analysis.get('text_length_analysis', {})
        if tla:
            print(f"\n📏 TEXT LENGTH ANALYSIS:")
            print(f"  Avg length (all): {tla.get('avg_length_all', 0):.1f} words")
            print(f"  Avg length (errors): {tla.get('avg_length_errors', 0):.1f} words")
            print(f"  Avg length (correct): {tla.get('avg_length_correct', 0):.1f} words")
        
        if categories:
            print(f"\n🔍 ERROR CATEGORIES:")
            for category, indices in categories.items():
                if indices:
                    print(f"  {category}: {len(indices)} examples")
        
        if content_patterns:
            print(f"\n📝 CONTENT PATTERNS IN ERRORS:")
            for pattern, indices in content_patterns.items():
                if indices:
                    print(f"  {pattern}: {len(indices)} examples")
        
        print(f"\n📝 TOP 5 MISCLASSIFIED EXAMPLES:")
        misclassified = self.get_misclassified_examples(5)
        if len(misclassified) > 0:
            for i, (idx, row) in enumerate(misclassified.iterrows(), 1):
                print(f"\n  Example {i} (True: {row['true_label']} → Pred: {row['predicted_label']}):")
                print(f"    Length: {row['text_length']} words")
                print(f"    Text: {str(row['text'])[:250]}...")


# ================================================================
# CROSS-MODEL COMMON ERROR ANALYSIS
# ================================================================

def find_common_misclassifications(model_outputs: Dict, 
                                   texts: List[str], 
                                   true_labels: List[int],
                                   n_examples: int = 5) -> List[int]:
    """
    Find examples misclassified by ALL models (common error patterns).
    """
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMMON ERROR ANALYSIS")
    print("Examples misclassified by ALL models")
    print(f"{'='*70}")
    
    # Find indices where ALL models made errors
    all_misclassified = None
    
    for model_name, outputs in model_outputs.items():
        predictions = np.array(outputs['predictions'])
        true = np.array(true_labels)
        misclassified = predictions != true
        
        if all_misclassified is None:
            all_misclassified = set(np.where(misclassified)[0])
        else:
            all_misclassified &= set(np.where(misclassified)[0])
    
    if not all_misclassified:
        print("✅ No examples misclassified by all models!")
        return []
    
    common_indices = sorted(list(all_misclassified))
    print(f"\n📌 {len(common_indices)} examples misclassified by ALL models")
    print(f"   ({len(common_indices)/len(true_labels)*100:.1f}% of test set)\n")
    
    # PATTERN 1: Text Length
    common_lengths = [len(texts[i].split()) for i in common_indices]
    all_lengths = [len(t.split()) for t in texts]
    
    print("📏 LENGTH PATTERN:")
    print(f"   Common errors avg length: {np.mean(common_lengths):.0f} words")
    print(f"   Overall avg length:       {np.mean(all_lengths):.0f} words")
    
    if np.mean(common_lengths) > np.mean(all_lengths) * 1.15:
        print(f"   → Common errors tend to be LONGER texts (harder to classify)")
    elif np.mean(common_lengths) < np.mean(all_lengths) * 0.85:
        print(f"   → Common errors tend to be SHORTER texts (less context)")
    else:
        print(f"   → No significant length bias")
    
    # PATTERN 2: Label Distribution
    common_true = [true_labels[i] for i in common_indices]
    label_counts = np.bincount(common_true)
    neg_count = label_counts[0] if len(label_counts) > 0 else 0
    pos_count = label_counts[1] if len(label_counts) > 1 else 0
    
    print(f"\n🏷️  LABEL PATTERN:")
    print(f"   Negative examples: {neg_count}")
    print(f"   Positive examples: {pos_count}")
    
    if neg_count > pos_count * 1.5:
        print(f"   → All models struggle more with NEGATIVE reviews")
    elif pos_count > neg_count * 1.5:
        print(f"   → All models struggle more with POSITIVE reviews")
    else:
        print(f"   → Balanced distribution")
    
    # PATTERN 3: Content Keywords
    sentiment_words = ['great', 'good', 'excellent', 'bad', 'terrible', 'worst',
                       'love', 'hate', 'not', "n't", 'but', 'however', 'although',
                       'boring', 'waste', 'best', 'awful', 'amazing']
    
    word_counts = Counter()
    for i in common_indices[:50]:
        text = texts[i].lower()
        for word in sentiment_words:
            if f' {word} ' in f' {text} ' or text.startswith(word) or text.endswith(word):
                word_counts[word] += 1
    
    if word_counts:
        print(f"\n📝 KEYWORD PATTERN (frequent in common errors):")
        for word, count in word_counts.most_common(8):
            bar = '█' * min(count, 20)
            print(f"   '{word}': {count:>3} {bar}")
    
    # PATTERN 4: Show concrete examples
    print(f"\n{'─'*60}")
    print(f"TOP {n_examples} COMMON MISCLASSIFIED EXAMPLES:")
    print(f"{'─'*60}")
    
    common_indices_sorted = sorted(common_indices, key=lambda i: len(texts[i].split()), reverse=True)
    
    for rank, idx in enumerate(common_indices_sorted[:n_examples], 1):
        text = texts[idx]
        true_label = true_labels[idx]
        
        print(f"\n  Example {rank} (Index: {idx}):")
        print(f"  True: {true_label} ({'Positive' if true_label == 1 else 'Negative'}) | "
              f"Length: {len(text.split())} words")
        
        # All model predictions
        for model_name, outputs in model_outputs.items():
            pred = outputs['predictions'][idx]
            correct = "✓" if pred == true_label else "✗"
            print(f"    {model_name:<20}: {pred} ({'Pos' if pred == 1 else 'Neg'}) {correct}")
        
        # Text preview
        text_preview = text[:300] + "..." if len(text) > 300 else text
        print(f"  Text: {text_preview}")
        
        # Why might it be hard?
        text_lower = text.lower()
        reasons = []
        if any(w in text_lower for w in ['but', 'however', 'although']):
            reasons.append("mixed sentiment (contrast words)")
        if text_lower.count('not') + text_lower.count("n't") >= 2:
            reasons.append("negation-heavy")
        if len(text.split()) > 300:
            reasons.append("very long")
        if len(text.split()) < 25:
            reasons.append("very short (little context)")
        if reasons:
            print(f"  Likely difficulty: {', '.join(reasons)}")
    
    return common_indices


# ================================================================
# MAIN ANALYSIS FUNCTIONS
# ================================================================

def analyze_all_models(model_outputs: Dict, 
                       texts: List[str], 
                       true_labels: List[int],
                       save_dir: str = RESULTS_DIR):
    """Analyze and compare errors across ALL models."""
    print("=" * 70)
    print("COMPREHENSIVE ERROR ANALYSIS ACROSS ALL MODELS")
    print("=" * 70)
    
    all_analyses = {}
    
    for model_name, outputs in model_outputs.items():
        print(f"\n{'='*70}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*70}")
        
        predictions = outputs['predictions']
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        analyzer = ErrorAnalyzer(
            texts=texts,
            true_labels=true_labels,
            predicted_labels=predictions,
            model_name=model_name
        )
        
        analyzer.print_report()
        
        cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
        analyzer.plot_confusion_matrix(save_path=cm_path)
        
        dist_path = os.path.join(save_dir, f'error_distribution_{model_name}.png')
        analyzer.plot_error_distribution(save_path=dist_path)
        
        csv_path = os.path.join(save_dir, f'misclassified_{model_name}.csv')
        analyzer.export_misclassified(csv_path, n_examples=10)
        
        all_analyses[model_name] = analyzer.analyze_error_patterns()
    
    # Save combined analysis
    combined_path = os.path.join(save_dir, 'q1_error_analysis_all_models.json')
    
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    
    with open(combined_path, 'w') as f:
        json.dump(convert(all_analyses), f, indent=4)
    print(f"\n✓ Combined analysis saved to {combined_path}")
    
    # Print cross-model comparison
    print_cross_model_comparison(all_analyses)
    
    # CROSS-MODEL COMMON ERRORS
    common_errors = find_common_misclassifications(model_outputs, texts, true_labels, n_examples=5)
    
    if common_errors:
        common_path = os.path.join(save_dir, 'common_misclassified.csv')
        common_df = pd.DataFrame({
            'index': common_errors,
            'text': [texts[i][:300] for i in common_errors],
            'true_label': [true_labels[i] for i in common_errors],
            'length': [len(texts[i].split()) for i in common_errors]
        })
        common_df.to_csv(common_path, index=False)
        print(f"✓ Common errors saved to {common_path}")
    
    return all_analyses


def print_cross_model_comparison(all_analyses: Dict):
    """Print comparison of error patterns across models."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL ERROR COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25}", end='')
    for model_name in all_analyses.keys():
        print(f"{model_name:<15}", end='')
    print()
    print('─' * 55)
    
    for metric in ['accuracy', 'error_rate', 'total_errors']:
        print(f"{metric:<25}", end='')
        for analysis in all_analyses.values():
            value = analysis.get(metric, 'N/A')
            if isinstance(value, float):
                print(f"{value:<15.4f}", end='')
            else:
                print(f"{str(value):<15}", end='')
        print()
    
    # FP/FN comparison
    if 'false_positives' in list(all_analyses.values())[0]:
        print(f"\n{'─'*55}")
        for error_type in ['false_positives', 'false_negatives']:
            print(f"{error_type:<25}", end='')
            for analysis in all_analyses.values():
                print(f"{analysis.get(error_type, 'N/A'):<15}", end='')
            print()
    
    best_model = max(all_analyses.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n{'='*70}")
    print(f"🏆 Best Performing Model: {best_model[0]} "
          f"(Accuracy: {best_model[1]['accuracy']:.4f})")
    print(f"{'='*70}")


def analyze_preprocessing_impact(preprocessing_results_path: str):
    """Analyze and visualize preprocessing impact."""
    if not os.path.exists(preprocessing_results_path):
        print(f"Preprocessing results not found at {preprocessing_results_path}")
        return
    
    with open(preprocessing_results_path, 'r') as f:
        pp_results = json.load(f)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING IMPACT VISUALIZATION")
    print(f"{'='*70}")
    
    configs = [r['configuration'].replace('_', ' ')[:30] for r in pp_results]
    accuracies = [r['accuracy'] for r in pp_results]
    f1_scores = [r['f1_macro'] for r in pp_results]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                   color='#4ecdc4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', 
                   color='#ff6b6b', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Preprocessing Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Preprocessing Impact on Model Performance (TF-IDF + LR)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0.75, 0.90)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'preprocessing_impact.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Preprocessing impact plot saved to {save_path}")
    plt.close()
    
    best = max(pp_results, key=lambda x: x['f1_macro'])
    worst = min(pp_results, key=lambda x: x['f1_macro'])
    
    print(f"\n📊 PREPROCESSING FINDINGS:")
    print(f"  Best: {best['configuration']} (Acc={best['accuracy']:.4f}, F1={best['f1_macro']:.4f})")
    print(f"  Worst: {worst['configuration']} (Acc={worst['accuracy']:.4f}, F1={worst['f1_macro']:.4f})")
    print(f"  Impact range: {best['f1_macro'] - worst['f1_macro']:.4f} F1 points")


def main():
    """Main function to run error analysis."""
    from datasets import load_dataset
    
    print("=" * 70)
    print("Q1 ERROR ANALYSIS")
    print("=" * 70)
    
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    test_data = dataset['test'].shuffle(seed=42).select(range(500))
    
    test_texts = test_data['text']
    test_labels = list(test_data['label'])
    
    predictions_path = os.path.join(RESULTS_DIR, 'q1_predictions.json')
    
    if os.path.exists(predictions_path):
        print(f"Loading saved predictions from {predictions_path}")
        with open(predictions_path, 'r') as f:
            saved_predictions = json.load(f)
        
        model_outputs = {}
        for model_name, pred_data in saved_predictions.items():
            predictions = pred_data['predictions']
            
            print(f"\n{model_name}:")
            print(f"  Pred distribution: {np.bincount(predictions)}")
            print(f"  True distribution: {np.bincount(test_labels)}")
            
            model_outputs[model_name] = {
                'predictions': predictions,
                'probabilities': None,
                'true_labels': test_labels
            }
        
        analyze_all_models(model_outputs, test_texts, test_labels)
    else:
        print("No saved predictions found. Run train.py first!")
    
    pp_results_path = os.path.join(RESULTS_DIR, 'q1_preprocessing_analysis.json')
    if os.path.exists(pp_results_path):
        analyze_preprocessing_impact(pp_results_path)
    
    print(f"\n{'='*70}")
    print("Q1 Error Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()