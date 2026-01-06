import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Dassl'))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import json

from dassl.utils import set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.hair_length
import datasets.hair_frizz
import datasets.hair_type

import trainers.coop
import trainers.cocoop
import trainers.zsclip


def extend_cfg(cfg):
    """Add new config variables."""
    from yacs.config import CfgNode as CN
    
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # Check if config exists in model directory (from training)
    import os.path as osp
    model_dir_abs = osp.abspath(args.model_dir)
    model_dir_basename = osp.basename(model_dir_abs)
    
    # Try to find config.yaml from training
    config_locations = []
    if model_dir_basename in ['prompt_learner', 'model', 'checkpoint', 'checkpoints']:
        # model-dir is a checkpoint subdirectory, go up one level
        config_locations.append(osp.join(osp.dirname(model_dir_abs), "config.yaml"))
    else:
        # model-dir is likely the run directory itself
        config_locations.append(osp.join(model_dir_abs, "config.yaml"))
        # Also try parent directory as fallback
        config_locations.append(osp.join(osp.dirname(model_dir_abs), "config.yaml"))
    
    model_config_file = None
    for loc in config_locations:
        if osp.exists(loc):
            model_config_file = loc
            break
    
    if model_config_file and not args.ignore_trained_config:
        print(f"Loading config from trained model: {model_config_file}")
        cfg.merge_from_file(model_config_file)
    else:
        # Fallback to provided config files
        if args.dataset_config_file:
            cfg.merge_from_file(args.dataset_config_file)
        if args.config_file:
            cfg.merge_from_file(args.config_file)
    
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg


def plot_confusion_matrix(cm, cm_normalized, class_names, output_dir, accuracy, threshold=None):
    """Plot and save confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create title with accuracy
    title_suffix = f" (Threshold: {threshold:.2f})" if threshold else ""
    suptitle = f'Overall Accuracy: {accuracy:.2%}{title_suffix}'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized * 100, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                cbar_kws={'label': 'Percentage (%)'})
    axes[0].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Raw counts confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                cbar_kws={'label': 'Count'})
    axes[1].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'confusion_matrix_threshold_{threshold:.2f}.png' if threshold else 'confusion_matrix.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_confidence_histogram(all_preds, all_labels, all_probs, output_dir):
    """Plot and save confidence histogram for predictions."""
    confidences = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels
    confidences_correct = confidences[correct_mask]
    confidences_incorrect = confidences[~correct_mask]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall confidence distribution
    axes[0, 0].hist(confidences, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidences.mean():.3f}')
    axes[0, 0].axvline(np.median(confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidences):.3f}')
    axes[0, 0].set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Confidence', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Correct predictions confidence
    axes[0, 1].hist(confidences_correct, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(confidences_correct.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {confidences_correct.mean():.3f}')
    axes[0, 1].set_title(f'Correct Predictions (n={len(confidences_correct)})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Incorrect predictions confidence
    axes[1, 0].hist(confidences_incorrect, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(confidences_incorrect.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {confidences_incorrect.mean():.3f}')
    axes[1, 0].set_title(f'Incorrect Predictions (n={len(confidences_incorrect)})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Comparison: Correct vs Incorrect
    axes[1, 1].hist(confidences_correct, bins=50, color='green', alpha=0.5, label='Correct', edgecolor='black')
    axes[1, 1].hist(confidences_incorrect, bins=50, color='red', alpha=0.5, label='Incorrect', edgecolor='black')
    axes[1, 1].set_title('Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confidence_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence histogram saved to {save_path}")
    
    # Print statistics
    print(f"\nConfidence Statistics:")
    print(f"  Overall Mean: {confidences.mean():.4f} ± {confidences.std():.4f}")
    print(f"  Overall Median: {np.median(confidences):.4f}")
    print(f"  Correct Predictions Mean: {confidences_correct.mean():.4f} ± {confidences_correct.std():.4f}")
    print(f"  Incorrect Predictions Mean: {confidences_incorrect.mean():.4f} ± {confidences_incorrect.std():.4f}")


def plot_precision_recall_per_class(precision, recall, f1, class_names, output_dir):
    """Plot precision, recall, and F1 scores per class."""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='steelblue')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='orange')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='green')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1-Score per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'precision_recall_per_class.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall per class plot saved to {save_path}")


def save_wrong_classification(img_path, true_label, pred_label, confidence, output_path):
    """Save image with prediction annotations."""
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    
    # Create a new image with extra space for text
    text_height = 100
    new_img = Image.new('RGB', (width, height + text_height), 'white')
    new_img.paste(img, (0, 0))
    
    # Add text
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    text = f"True: {true_label}\nPredicted: {pred_label}\nConfidence: {confidence:.2%}"
    
    # Draw text with background
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height + 10
    
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    new_img.save(output_path)


def compute_metrics_with_threshold(all_preds, all_labels, all_probs, confidence_threshold):
    """Compute metrics for predictions above confidence threshold."""
    confidences = np.max(all_probs, axis=1)
    threshold_mask = confidences >= confidence_threshold
    
    if threshold_mask.sum() == 0:
        return None, 0
    
    filtered_preds = all_preds[threshold_mask]
    filtered_labels = all_labels[threshold_mask]
    filtered_probs = all_probs[threshold_mask]
    
    accuracy = accuracy_score(filtered_labels, filtered_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_labels, filtered_preds, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        filtered_labels, filtered_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(filtered_labels, filtered_preds)
    cm_normalized = confusion_matrix(filtered_labels, filtered_preds, normalize='true')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'n_samples': threshold_mask.sum(),
        'filtered_preds': filtered_preds,
        'filtered_labels': filtered_labels,
        'filtered_probs': filtered_probs
    }, threshold_mask.sum()


def evaluate_split(trainer, data_loader, class_names, num_classes, split_name, output_dir, args):
    """Evaluate on a specific data split."""
    print(f"\n{'='*80}")
    print(f"Evaluating on {split_name.upper()} set...")
    print(f"{'='*80}")
    
    # Create split-specific output directory
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            images = batch["img"].to(trainer.device)
            labels = batch["label"].to(trainer.device)
            image_paths = batch["impath"]
            
            outputs = trainer.model_inference(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_image_paths.extend(image_paths)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {split_name.upper()}")
    print(f"{'='*80}")
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes)
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    
    # Per-class results
    print(f"\nPer-class metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}")
    print("-" * 82)
    
    per_class_results = {}
    for i in range(num_classes):
        class_name = class_names[i]
        class_mask = all_labels == i
        class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() if class_mask.sum() > 0 else 0
        
        print(f"{class_name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10} {class_acc:>10.4f}")
        
        per_class_results[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
            'accuracy': float(class_acc)
        }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')
    
    print(f"\nConfusion Matrix (raw counts):")
    print(cm)
    
    # Save confusion matrix
    plot_confusion_matrix(cm, cm_normalized, class_names, split_output_dir, accuracy)
    
    # Plot precision-recall per class
    plot_precision_recall_per_class(precision, recall, f1, class_names, split_output_dir)
    
    # Plot confidence histogram
    plot_confidence_histogram(all_preds, all_labels, all_probs, split_output_dir)
    
    # Compute confidence statistics
    confidences = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels
    confidences_correct = confidences[correct_mask]
    confidences_incorrect = confidences[~correct_mask]
    
    # Save numerical results
    results = {
        'overall_accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'per_class_results': per_class_results,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names,
        'confidence_statistics': {
            'overall_mean': float(confidences.mean()),
            'overall_std': float(confidences.std()),
            'overall_median': float(np.median(confidences)),
            'correct_mean': float(confidences_correct.mean()),
            'correct_std': float(confidences_correct.std()),
            'incorrect_mean': float(confidences_incorrect.mean()),
            'incorrect_std': float(confidences_incorrect.std())
        }
    }
    
    results_file = os.path.join(split_output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Evaluate with confidence threshold if provided
    if args.confidence_threshold is not None and args.confidence_threshold > 0:
        print(f"\n{'='*80}")
        print(f"EVALUATION WITH CONFIDENCE THRESHOLD: {args.confidence_threshold}")
        print(f"{'='*80}")
        
        threshold_metrics, n_samples = compute_metrics_with_threshold(
            all_preds, all_labels, all_probs, args.confidence_threshold
        )
        
        if threshold_metrics is None or n_samples == 0:
            print(f"No predictions above confidence threshold {args.confidence_threshold}")
        else:
            print(f"\nFiltered samples: {n_samples} / {len(all_preds)} ({n_samples/len(all_preds)*100:.1f}%)")
            print(f"Accuracy (threshold {args.confidence_threshold}): {threshold_metrics['accuracy']:.4f} ({threshold_metrics['accuracy']*100:.2f}%)")
            print(f"\nMacro-averaged metrics (threshold {args.confidence_threshold}):")
            print(f"  Precision: {threshold_metrics['precision_macro']:.4f}")
            print(f"  Recall:    {threshold_metrics['recall_macro']:.4f}")
            print(f"  F1-Score:  {threshold_metrics['f1_macro']:.4f}")
            
            # Per-class results with threshold
            print(f"\nPer-class metrics (threshold {args.confidence_threshold}):")
            print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print("-" * 72)
            
            for i in range(num_classes):
                if i < len(threshold_metrics['precision']):
                    print(f"{class_names[i]:<20} {threshold_metrics['precision'][i]:>10.4f} "
                          f"{threshold_metrics['recall'][i]:>10.4f} {threshold_metrics['f1'][i]:>10.4f} "
                          f"{threshold_metrics['support'][i]:>10}")
            
            # Save threshold confusion matrix
            plot_confusion_matrix(
                threshold_metrics['confusion_matrix'],
                threshold_metrics['confusion_matrix_normalized'],
                class_names,
                split_output_dir,
                threshold_metrics['accuracy'],
                threshold=args.confidence_threshold
            )
            
            # Save threshold results to JSON
            threshold_results = {
                'confidence_threshold': args.confidence_threshold,
                'n_samples_above_threshold': int(n_samples),
                'percentage_above_threshold': float(n_samples/len(all_preds)),
                'accuracy': float(threshold_metrics['accuracy']),
                'macro_precision': float(threshold_metrics['precision_macro']),
                'macro_recall': float(threshold_metrics['recall_macro']),
                'macro_f1': float(threshold_metrics['f1_macro']),
                'per_class_precision': threshold_metrics['precision'].tolist(),
                'per_class_recall': threshold_metrics['recall'].tolist(),
                'per_class_f1': threshold_metrics['f1'].tolist(),
                'per_class_support': threshold_metrics['support'].tolist(),
                'confusion_matrix': threshold_metrics['confusion_matrix'].tolist(),
                'confusion_matrix_normalized': threshold_metrics['confusion_matrix_normalized'].tolist()
            }
            
            threshold_results_file = os.path.join(
                split_output_dir,
                f'evaluation_results_threshold_{args.confidence_threshold:.2f}.json'
            )
            with open(threshold_results_file, 'w') as f:
                json.dump(threshold_results, f, indent=2)
            print(f"\nThreshold results saved to {threshold_results_file}")
    
    # Save wrong classifications
    if args.save_wrong_classifications:
        print(f"\nSaving misclassified images...")
        wrong_dir = os.path.join(split_output_dir, 'wrong_classifications')
        os.makedirs(wrong_dir, exist_ok=True)
        
        wrong_count = 0
        for i, (pred, label, prob, img_path) in enumerate(zip(all_preds, all_labels, all_probs, all_image_paths)):
            if pred != label:
                confidence = prob[pred]
                true_class = class_names[label]
                pred_class = class_names[pred]
                
                # Create filename
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_name = f"{wrong_count:04d}_true_{true_class}_pred_{pred_class}_conf_{confidence:.2f}{ext}"
                output_path = os.path.join(wrong_dir, output_name)
                
                save_wrong_classification(img_path, true_class, pred_class, confidence, output_path)
                wrong_count += 1
        
        print(f"Saved {wrong_count} misclassified images to {wrong_dir}")
        
        # Save wrong classifications summary
        wrong_summary = []
        for i, (pred, label, prob, img_path) in enumerate(zip(all_preds, all_labels, all_probs, all_image_paths)):
            if pred != label:
                wrong_summary.append({
                    'image_path': img_path,
                    'true_label': class_names[label],
                    'predicted_label': class_names[pred],
                    'confidence': float(prob[pred]),
                    'true_label_confidence': float(prob[label])
                })
        
        wrong_summary_file = os.path.join(split_output_dir, 'wrong_classifications_summary.json')
        with open(wrong_summary_file, 'w') as f:
            json.dump(wrong_summary, f, indent=2)
        print(f"Wrong classifications summary saved to {wrong_summary_file}")


def evaluate_model(args):
    """Main evaluation function."""
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    
    # Create output directory
    os.makedirs(args.eval_output_dir, exist_ok=True)
    
    print(f"\nTrainer: {cfg.TRAINER.NAME}")
    print(f"N_CTX: {cfg.TRAINER.COOP.N_CTX if hasattr(cfg.TRAINER, 'COOP') else 'N/A'}")
    print(f"CSC: {cfg.TRAINER.COOP.CSC if hasattr(cfg.TRAINER, 'COOP') else 'N/A'}")
    print(f"Evaluating model from: {args.model_dir}")
    print(f"Output directory: {args.eval_output_dir}")
    
    # Build trainer and load model
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.set_model_mode("eval")
    
    # Get class names
    lab2cname = trainer.dm.lab2cname
    class_names = [lab2cname[i] for i in sorted(lab2cname.keys())]
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Determine which splits to evaluate
    splits_to_evaluate = []
    if 'val' in args.eval_split or 'both' in args.eval_split:
        if trainer.val_loader is not None:
            splits_to_evaluate.append(('val', trainer.val_loader))
        else:
            print("Warning: Validation loader not available")
    
    if 'test' in args.eval_split or 'both' in args.eval_split:
        splits_to_evaluate.append(('test', trainer.test_loader))
    
    if not splits_to_evaluate:
        print("Error: No valid splits to evaluate")
        return
    
    # Evaluate each split
    for split_name, data_loader in splits_to_evaluate:
        evaluate_split(trainer, data_loader, class_names, num_classes, split_name, args.eval_output_dir, args)
    
    print("\n" + "="*80)
    print("All evaluations complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation script for multi-class classification")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--eval-output-dir", type=str, default="", help="directory to save evaluation results (default: creates 'evaluation' folder in model parent directory)")
    parser.add_argument("--model-dir", type=str, required=True, help="directory containing the model checkpoint (e.g., path/to/prompt_learner)")
    parser.add_argument("--load-epoch", type=int, default=None, help="specific epoch to load (default: best model)")
    parser.add_argument("--seed", type=int, default=-1, help="random seed")
    parser.add_argument("--config-file", type=str, default="", help="path to config file (optional if config.yaml exists in model dir)")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to dataset config file")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--eval-split", type=str, nargs='+', default=['test'], help="which split(s) to evaluate: 'val', 'test', or 'both'")
    parser.add_argument("--save-wrong-classifications", action="store_true", help="save images of wrong classifications with annotations")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="confidence threshold for filtering predictions (e.g., 0.7)")
    parser.add_argument("--ignore-trained-config", action="store_true", help="ignore config.yaml from training, use provided configs instead")
    parser.add_argument("--output-dir", type=str, default="", help="trainer output directory (for compatibility)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    
    args = parser.parse_args()
    
    # Auto-generate eval-output-dir if not provided
    if not args.eval_output_dir:
        import os.path as osp
        model_dir_abs = osp.abspath(args.model_dir)
        
        # Check if model-dir ends with a checkpoint subdirectory (e.g., prompt_learner)
        model_dir_basename = osp.basename(model_dir_abs)
        if model_dir_basename in ['prompt_learner', 'model', 'checkpoint', 'checkpoints']:
            # Go up one level to the training run directory
            run_dir = osp.dirname(model_dir_abs)
        else:
            # Already at the training run directory
            run_dir = model_dir_abs
        
        args.eval_output_dir = osp.join(run_dir, "evaluation")
        print(f"No --eval-output-dir provided, using: {args.eval_output_dir}")
    
    evaluate_model(args)

