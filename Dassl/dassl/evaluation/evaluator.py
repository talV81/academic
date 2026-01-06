import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            # Compute normalized confusion matrix
            cmat_normalized = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            # Compute raw counts confusion matrix
            cmat_counts = confusion_matrix(
                self._y_true, self._y_pred, normalize=None
            )
            
            # Save as PyTorch tensor
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat_normalized, save_path)
            print(f"Confusion matrix is saved to {save_path}")
            
            # Create and save plot
            self._plot_confusion_matrix(
                cmat_normalized, cmat_counts, 
                self._lab2cname, 
                self.cfg.OUTPUT_DIR
            )

        return results
    
    def _plot_confusion_matrix(self, cmat_normalized, cmat_counts, lab2cname, output_dir):
        """Plot and save confusion matrix."""
        if lab2cname is None:
            # Create default labels if lab2cname not available
            num_classes = cmat_normalized.shape[0]
            class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            # Get class names in order
            labels = sorted(lab2cname.keys())
            class_names = [lab2cname[label] for label in labels]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Normalized confusion matrix (percentages)
        sns.heatmap(
            cmat_normalized * 100,  # Convert to percentage
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            cbar_kws={'label': 'Percentage (%)'}
        )
        axes[0].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 2: Raw counts confusion matrix
        sns.heatmap(
            cmat_counts,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            cbar_kws={'label': 'Count'}
        )
        axes[1].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = osp.join(output_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot is saved to {plot_path}")