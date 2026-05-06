"""
Utility modules for CENG467 NLP Project.
Contains shared metrics and visualization functions.
"""

from .metrics import (
    MetricsTracker,
    compute_multiclass_metrics,
    format_metrics,
    calculate_perplexity,
    sequence_accuracy,
    count_parameters,
    format_param_count,
    early_stopping_check
)

from .visualization import (
    set_plotting_style,
    plot_training_history,
    plot_confusion_matrix_heatmap,
    plot_model_comparison,
    plot_ner_error_analysis,
    plot_rouge_comparison,
    plot_attention_weights,
    plot_perplexity_comparison,
    plot_text_length_distribution,
    save_all_figures
)

__all__ = [
    # Metrics
    'MetricsTracker',
    'compute_multiclass_metrics',
    'format_metrics',
    'calculate_perplexity',
    'sequence_accuracy',
    'count_parameters',
    'format_param_count',
    'early_stopping_check',
    # Visualization
    'set_plotting_style',
    'plot_training_history',
    'plot_confusion_matrix_heatmap',
    'plot_model_comparison',
    'plot_ner_error_analysis',
    'plot_rouge_comparison',
    'plot_attention_weights',
    'plot_perplexity_comparison',
    'plot_text_length_distribution',
    'save_all_figures'
]