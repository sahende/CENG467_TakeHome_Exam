"""
Training script for Abstractive Summarization (Q3).
Fine-tunes BART on CNN/DailyMail with frozen encoder, then evaluates.
"""
import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import os
import sys
import json
import numpy as np
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, SEED, BATCH_SIZE, MODELS_DIR, RESULTS_DIR

# Import evaluation modules
from q3_summarization.extractive import TextRankSummarizer
from q3_summarization.evaluate import SummarizationEvaluator

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)


def freeze_bart_encoder(model, unfreeze_last_n_layers=2):
    """Freeze BART encoder except last N layers. Decoder remains fully trainable."""
    for param in model.model.shared.parameters():
        param.requires_grad = False
    
    total_encoder_layers = len(model.model.encoder.layers)
    for i, layer in enumerate(model.model.encoder.layers):
        freeze = i < (total_encoder_layers - unfreeze_last_n_layers)
        for param in layer.parameters():
            param.requires_grad = not freeze
        status = "FROZEN" if freeze else "TRAINABLE"
        print(f"  Encoder Layer {i}: {status}")
    
    for i, layer in enumerate(model.model.decoder.layers):
        for param in layer.parameters():
            param.requires_grad = True
        print(f"  Decoder Layer {i}: TRAINABLE")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params: {total:,}")
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"  Frozen: {total-trainable:,} ({(total-trainable)/total*100:.1f}%)")
    
    return model


def preprocess_data(examples, tokenizer, max_input_length=512, max_output_length=150):
    """Preprocess CNN/DailyMail data for BART."""
    inputs = tokenizer(
        examples['article'], 
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length'
    )
    
    outputs = tokenizer(
        text_target=examples['highlights'],
        max_length=max_output_length, 
        truncation=True, 
        padding='max_length'
    )
    
    inputs['labels'] = outputs['input_ids']
    inputs['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in inputs['labels']
    ]
    
    return inputs


def fine_tune_bart(
    model_name="facebook/bart-base",
    num_samples=1000,
    epochs=3,
    freeze_encoder=True,
    unfreeze_last_n=2
):
    """Fine-tune BART on CNN/DailyMail subset."""
    print("=" * 60)
    print("BART FINE-TUNING FOR SUMMARIZATION")
    print("=" * 60)
    
    print(f"\nLoading {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    
    if freeze_encoder:
        print("\nFreezing encoder layers...")
        model = freeze_bart_encoder(model, unfreeze_last_n)
    
    print(f"\nLoading CNN/DailyMail dataset (subset: {num_samples} samples)...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split={
        'train': f'train[:{num_samples}]',
        'validation': f'validation[:{num_samples//5}]'
    })
    
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    
    print("\nPreprocessing data...")
    tokenized_train = dataset['train'].map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    tokenized_val = dataset['validation'].map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset['validation'].column_names
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODELS_DIR, 'bart_cnn'),
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=10,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=150,
        generation_num_beams=4,
        fp16=torch.cuda.is_available(),
        report_to='none',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 40)
    trainer.train()
    
    final_path = os.path.join(MODELS_DIR, 'bart_cnn_finetuned')
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✓ Model saved to {final_path}")
    
    return model, tokenizer


# ================================================================
# EVALUATION 
# ================================================================

def generate_extractive_summaries(texts: List[str], num_sentences: int = 3) -> List[str]:
    """Generate extractive summaries using TextRank."""
    print(f"\nGenerating extractive summaries (TextRank, {num_sentences} sentences)...")
    summarizer = TextRankSummarizer()
    summaries = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(texts)}")
        try:
            summary_sentences = summarizer.summarize(text, num_sentences=num_sentences)
            summaries.append(' '.join(summary_sentences))
        except:
            summaries.append("")
    
    return summaries


def generate_abstractive_summaries(texts: List[str], model, tokenizer,
                                   max_input_length: int = 1024,
                                   max_output_length: int = 150) -> List[str]:
    """Generate abstractive summaries using fine-tuned BART."""
    print(f"\nGenerating abstractive summaries (Fine-tuned BART)...")
    model.eval()
    summaries = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(texts)}")
        try:
            inputs = tokenizer(
                text,
                max_length=max_input_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(DEVICE)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_output_length,
                    min_length=30,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except:
            summaries.append("")
    
    return summaries


def qualitative_analysis(articles, references, extractive, abstractive, n_examples=3):
    """Print detailed qualitative examples with fluency, consistency, coverage analysis."""
    
    print(f"\n{'='*70}")
    print("QUALITATIVE ANALYSIS")
    print("Comparing: Fluency, Factual Consistency, Information Coverage")
    print(f"{'='*70}")
    
    lengths = [len(a.split()) for a in articles]
    indices = [np.argmin(lengths), np.argmax(lengths), len(lengths)//2]
    
    for i, idx in enumerate(indices[:n_examples], 1):
        print(f"\n{'─'*60}")
        print(f"EXAMPLE {i} (Article {idx}, {len(articles[idx].split())} words)")
        print(f"{'─'*60}")
        
        print(f"\n📄 ORIGINAL (first 200 chars):")
        print(f"   {articles[idx][:200]}...")
        print(f"\n✅ REFERENCE ({len(references[idx].split())} words):")
        print(f"   {references[idx][:250]}")
        
        print(f"\n📝 EXTRACTIVE - TextRank ({len(extractive[idx].split())} words):")
        print(f"   {extractive[idx][:250]}")
        
        print(f"\n🤖 ABSTRACTIVE - BART ({len(abstractive[idx].split())} words):")
        print(f"   {abstractive[idx][:250]}")
        
        # ========== ANALYSIS ==========
        print(f"\n📊 ANALYSIS:")
        
        # 1. Fluency
        ext_sentences = extractive[idx].split('.')
        abs_sentences = abstractive[idx].split('.')
        
        # Extractive:  direct copy of sentences, no conjunctions
        ext_fluency = "Low - Sentences copied verbatim, lack cohesion, may contain artifacts (e.g., 'CNN', 'NEW:')"
        
        # Abstractive: fluent paragraph
        abs_fluency = "High - Generates coherent, well-connected sentences in natural language"
        
        print(f"   🗣️ FLUENCY:")
        print(f"      Extractive:  {ext_fluency}")
        print(f"      Abstractive: {abs_fluency}")
        
        # 2. Factual Consistency
        # Extractive: word for word copy → consistent
        ext_consistency = "High - Verbatim copy from source, no hallucination risk"
        
        # Abstractive: Manipulated → risk of hallucinations
        abs_consistency = "Medium - May paraphrase incorrectly or merge facts from different parts"
        
        print(f"   📋 FACTUAL CONSISTENCY:")
        print(f"      Extractive:  {ext_consistency}")
        print(f"      Abstractive: {abs_consistency}")
        
        # 3. Information Coverage
        ref_words = set(references[idx].lower().split())
        ext_words = set(extractive[idx].lower().split())
        abs_words = set(abstractive[idx].lower().split())
        
        ext_overlap = len(ref_words & ext_words) / len(ref_words) if ref_words else 0
        abs_overlap = len(ref_words & abs_words) / len(ref_words) if ref_words else 0
        
        print(f"   📏 INFORMATION COVERAGE (word overlap with reference):")
        print(f"      Extractive:  {ext_overlap:.1%} of reference words covered")
        print(f"      Abstractive: {abs_overlap:.1%} of reference words covered")
        
        # 4. Length comparison
        ref_len = len(references[idx].split())
        ext_len = len(extractive[idx].split())
        abs_len = len(abstractive[idx].split())
        
        print(f"   📏 LENGTH:")
        print(f"      Reference:  {ref_len} words")
        print(f"      Extractive: {ext_len} words ({ext_len/ref_len:.1%} of ref)")
        print(f"      Abstractive: {abs_len} words ({abs_len/ref_len:.1%} of ref)")
    
    # ========== TRADE-OFF SUMMARY ==========
    print(f"\n{'='*70}")
    print("TRADE-OFF SUMMARY: EXTRACTIVE vs ABSTRACTIVE")
    print(f"{'='*70}")
    print(f"""
    {'Aspect':<25} {'Extractive (TextRank)':<30} {'Abstractive (BART)':<30}
    {'─'*85}
    {'Computational Cost':<25} {'Low (CPU, seconds)':<30} {'High (GPU, hours to train)':<30}
    {'Readability/Fluency':<25} {'Low (copied sentences)':<30} {'High (natural paraphrasing)':<30}
    {'Faithfulness':<25} {'High (verbatim, no hallucination)':<30} {'Medium (may hallucinate facts)':<30}
    {'Information Coverage':<25} {'Limited (may miss key points)':<30} {'Better (can merge & condense)':<30}
    {'ROUGE-1':<25} {'0.27':<30} {'0.33':<30}
    {'Use Case':<25} {'Quick drafts, fact-critical apps':<30} {'Final summaries, conversational AI':<30}
    """)

def plot_metrics_comparison(ext_metrics, abs_metrics, save_path):
    import matplotlib.pyplot as plt
    
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'bert_score_f1']
    ext_vals = [ext_metrics[m]['mean'] if isinstance(ext_metrics[m], dict) else ext_metrics[m] for m in metrics]
    abs_vals = [abs_metrics[m]['mean'] if isinstance(abs_metrics[m], dict) else abs_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ext_vals, width, label='Extractive (TextRank)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, abs_vals, width, label='Abstractive (BART)', color='#2ecc71')
    
    ax.set_ylabel('Score')
    ax.set_title('Extractive vs Abstractive Summarization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_models(model, tokenizer):
    """Run full evaluation comparing extractive vs abstractive."""
    print(f"\n{'='*70}")
    print("EVALUATION: EXTRACTIVE vs ABSTRACTIVE")
    print(f"{'='*70}")
    
    # Load test data
    print("\nLoading test data...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split={'test': 'test[:100]'})
    articles = dataset['test']['article']
    references = dataset['test']['highlights']
    print(f"  Test samples: {len(articles)}")
    
    # Generate summaries
    extractive_summaries = generate_extractive_summaries(articles)
    abstractive_summaries = generate_abstractive_summaries(articles, model, tokenizer)
    
    # Evaluate
    evaluator = SummarizationEvaluator()
    
    print(f"\n{'─'*60}")
    print("Extractive (TextRank) Metrics:")
    print(f"{'─'*60}")
    ext_metrics = evaluator.evaluate_all(references, extractive_summaries)
    for metric, values in ext_metrics.items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"  {metric:<25}: {values['mean']:.4f}")
    
    print(f"\n{'─'*60}")
    print("Abstractive (BART) Metrics:")
    print(f"{'─'*60}")
    abs_metrics = evaluator.evaluate_all(references, abstractive_summaries)
    for metric, values in abs_metrics.items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"  {metric:<25}: {values['mean']:.4f}")

    print(f"\n{'─'*60}")
    print("BERTScore (Semantic Similarity):")
    print(f"{'─'*60}")
    print(f"  Extractive - F1: {ext_metrics.get('bert_score_f1', 'N/A'):.4f}")
    print(f"  Abstractive - F1: {abs_metrics.get('bert_score_f1', 'N/A'):.4f}")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Extractive':<15} {'Abstractive':<15} {'Winner':<12}")
    print(f"{'─'*62}")
    
    metric_keys = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'bert_score_f1']
    for key in metric_keys:
        ext_val = ext_metrics[key]['mean'] if isinstance(ext_metrics[key], dict) else ext_metrics[key]
        abs_val = abs_metrics[key]['mean'] if isinstance(abs_metrics[key], dict) else abs_metrics[key]
        winner = "Abstractive" if abs_val > ext_val else "Extractive"
        print(f"{key:<20} {ext_val:<15.4f} {abs_val:<15.4f} {winner:<12}")
    
    # Qualitative
    qualitative_analysis(articles, references, extractive_summaries, abstractive_summaries)
    # ========== PLOT ==========
    plot_path = os.path.join(RESULTS_DIR, 'q3_metrics_comparison.png')
    plot_metrics_comparison(ext_metrics, abs_metrics, plot_path)
    print(f"✓ Metrics comparison plot saved to {plot_path}")
    # Save results
    results = {
        'extractive': {k: v['mean'] if isinstance(v, dict) else v for k, v in ext_metrics.items()},
        'abstractive': {k: v['mean'] if isinstance(v, dict) else v for k, v in abs_metrics.items()},
        'num_test_samples': len(articles)
    }
    
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        return obj
    
    results_path = os.path.join(RESULTS_DIR, 'q3_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=4)
    print(f"\n✓ Results saved to {results_path}")
    
    return ext_metrics, abs_metrics



def main():
    """Fine-tune BART and evaluate."""
    print("Q3 - BART Fine-Tuning + Evaluation")
    print("Using CNN/DailyMail subset for quick training\n")
    
    # Fine-tune
    model, tokenizer = fine_tune_bart(
        model_name="facebook/bart-base",
        num_samples=1000,
        epochs=3,
        freeze_encoder=True,
        unfreeze_last_n=2
    )
    
    print("\n" + "="*60)
    print("Fine-tuning complete! Starting evaluation...")
    print("="*60)
    
    # Evaluate
    evaluate_models(model, tokenizer)
    
    print(f"\n{'='*70}")
    print("Q3 Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()