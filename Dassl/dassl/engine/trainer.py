import time
import numpy as np
import os.path as osp
import datetime
import shutil
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        # Initialize device with error handling for NVML issues
        if cfg.USE_CUDA:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    # Try to get current device to verify it works (without NVML)
                    try:
                        _ = torch.cuda.current_device()
                    except Exception:
                        # If current_device fails, still use CUDA but skip device queries
                        print("Warning: CUDA device query failed, but CUDA is available. Continuing...")
                else:
                    print("Warning: CUDA not available, using CPU")
                    self.device = torch.device("cpu")
            except Exception as e:
                print(f"Warning: CUDA initialization failed (NVML issue?): {e}")
                print("Falling back to CPU...")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        # Check device count with error handling for NVML issues
        # Skip multi-GPU detection if NVML fails (common in containers/VMs)
        try:
            # Use a simple check - if this hangs, we'll skip it
            if self.device.type == 'cuda':
                # Try to get device count, but don't hang if NVML fails
                import threading
                device_count_result = [None]
                exception_occurred = [False]
                
                def get_device_count():
                    try:
                        device_count_result[0] = torch.cuda.device_count()
                    except Exception as e:
                        exception_occurred[0] = True
                        device_count_result[0] = 1  # Default to single GPU
                
                thread = threading.Thread(target=get_device_count)
                thread.daemon = True
                thread.start()
                thread.join(timeout=2.0)  # 2 second timeout
                
                if thread.is_alive():
                    print("Warning: GPU device count query timed out (NVML issue). Using single GPU.")
                    device_count = 1
                elif exception_occurred[0]:
                    print("Warning: GPU device count query failed (NVML issue). Using single GPU.")
                    device_count = 1
                else:
                    device_count = device_count_result[0] or 1
                
                if device_count > 1:
                    print(f"Detected {device_count} GPUs (use nn.DataParallel)")
                    self.model = nn.DataParallel(self.model)
        except Exception as e:
            print(f"Warning: Could not detect GPU count (NVML issue?): {e}")
            print("Continuing with single GPU assumption...")

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        
        # Check if we should resume from checkpoint
        if self.cfg.TRAIN.RESUME_FROM_CHECKPOINT:
            self.start_epoch = self.resume_model_if_exist(directory)
        else:
            print("TRAIN.RESUME_FROM_CHECKPOINT is False, starting from scratch")
            self.start_epoch = 0

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                # Check if best model exists, if not create it from last epoch
                names = self.get_model_names()
                best_model_path = osp.join(self.output_dir, names[0], "model-best.pth.tar")
                if not osp.exists(best_model_path):
                    print("Warning: model-best.pth.tar not found. Creating it from last epoch checkpoint.")
                    # Load last checkpoint and save as best
                    checkpoint_file = osp.join(self.output_dir, names[0], "checkpoint")
                    if osp.exists(checkpoint_file):
                        with open(checkpoint_file, "r") as f:
                            last_checkpoint = f.readline().strip()
                        last_checkpoint_path = osp.join(self.output_dir, names[0], last_checkpoint)
                        shutil.copy(last_checkpoint_path, best_model_path)
                        print(f"Created model-best.pth.tar from {last_checkpoint}")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()
        
        # Generate and save training plots
        self._save_training_plots()

    def _save_training_plots(self):
        """Generate and save training plots from TensorBoard logs."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            tb_log_dir = osp.join(self.output_dir, "tensorboard")
            if not osp.exists(tb_log_dir):
                print("No TensorBoard logs found, skipping plot generation")
                return
            
            # Load TensorBoard data
            event_acc = EventAccumulator(tb_log_dir)
            event_acc.Reload()
            
            available_tags = event_acc.Tags()['scalars']
            if not available_tags:
                print("No scalar data found in TensorBoard logs")
                return
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Helper function to get data from TensorBoard
            def get_metric_data(tag):
                if tag in available_tags:
                    scalar_events = event_acc.Scalars(tag)
                    steps = [s.step for s in scalar_events]
                    values = [s.value for s in scalar_events]
                    return steps, values
                return None, None
            
            # Helper function to aggregate per-batch metrics to per-epoch
            def aggregate_by_epoch(steps, values, num_batches):
                """Aggregate per-batch data to per-epoch averages."""
                if not steps or not values:
                    return None, None
                
                epoch_values = {}
                for step, value in zip(steps, values):
                    epoch = step // num_batches
                    if epoch not in epoch_values:
                        epoch_values[epoch] = []
                    epoch_values[epoch].append(value)
                
                epochs = sorted(epoch_values.keys())
                avg_values = [np.mean(epoch_values[e]) for e in epochs]
                return epochs, avg_values
            
            # Determine number of batches per epoch from training data
            train_loss_steps, train_loss_values = get_metric_data('train/loss')
            num_batches = self.num_batches if hasattr(self, 'num_batches') else None
            
            # If we can't determine num_batches, estimate from the data
            if num_batches is None and train_loss_steps:
                # Find the step gap that repeats (indicates epoch boundary)
                val_loss_steps, _ = get_metric_data('val/loss')
                if val_loss_steps and len(val_loss_steps) > 1:
                    num_batches = val_loss_steps[1] - val_loss_steps[0]
                else:
                    num_batches = len(train_loss_steps) // max(1, (self.max_epoch if hasattr(self, 'max_epoch') else 50))
            
            # Plot 1: Loss (train and validation)
            ax1 = axes[0]
            val_loss_steps, val_loss_values = get_metric_data('val/loss')
            
            if train_loss_steps and num_batches:
                train_epochs, train_loss_avg = aggregate_by_epoch(train_loss_steps, train_loss_values, num_batches)
                ax1.plot(train_epochs, train_loss_avg, linewidth=2, label='Train Loss', color='blue', marker='o', markersize=4)
            
            if val_loss_steps:
                ax1.plot(val_loss_steps, val_loss_values, linewidth=2, label='Val Loss', color='orange', marker='s', markersize=4)
            
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Plot 2: Accuracy (train and val) + Macro F1 (val)
            ax2 = axes[1]
            train_acc_steps, train_acc_values = get_metric_data('train/acc')
            val_acc_steps, val_acc_values = get_metric_data('val/accuracy')
            val_f1_steps, val_f1_values = get_metric_data('val/macro_f1')
            
            if train_acc_steps and num_batches:
                train_epochs, train_acc_avg = aggregate_by_epoch(train_acc_steps, train_acc_values, num_batches)
                ax2.plot(train_epochs, train_acc_avg, linewidth=2, label='Train Accuracy', color='blue', marker='o', markersize=4)
            
            if val_acc_steps:
                ax2.plot(val_acc_steps, val_acc_values, linewidth=2, label='Val Accuracy', color='orange', marker='s', markersize=4)
            if val_f1_steps:
                ax2.plot(val_f1_steps, val_f1_values, linewidth=2, label='Val Macro F1', color='green', linestyle='--', marker='^', markersize=4)
            
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Percentage (%)', fontsize=11)
            ax2.set_title('Accuracy & Macro F1', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Plot 3: Learning Rate (take last value per epoch)
            ax3 = axes[2]
            lr_steps, lr_values = get_metric_data('train/lr')
            
            if lr_steps and num_batches:
                # Take the last LR value of each epoch
                lr_epochs, lr_avg = aggregate_by_epoch(lr_steps, lr_values, num_batches)
                # For LR, use the last value instead of average
                epoch_lr = {}
                for step, value in zip(lr_steps, lr_values):
                    epoch = step // num_batches
                    if epoch not in epoch_lr or step > epoch_lr[epoch][0]:
                        epoch_lr[epoch] = (step, value)
                epochs = sorted(epoch_lr.keys())
                lr_values_per_epoch = [epoch_lr[e][1] for e in epochs]
                
                ax3.plot(epochs, lr_values_per_epoch, linewidth=2, label='Learning Rate', color='red', marker='o', markersize=4)
                ax3.set_xlabel('Epoch', fontsize=11)
                ax3.set_ylabel('Learning Rate', fontsize=11)
                ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=10)
                # Use scientific notation for y-axis if values are small
                ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            else:
                ax3.text(0.5, 0.5, 'No LR data available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            
            # Overall title and layout
            plt.suptitle(f'Training Summary - {osp.basename(self.output_dir)}', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save plot
            plot_path = osp.join(self.output_dir, "training_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training plots saved to {plot_path}")
            
        except ImportError:
            print("⚠️  TensorBoard not available for plot generation. Install with: pip install tensorboard")
        except Exception as e:
            print(f"⚠️  Failed to generate training plots: {e}")

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # Always evaluate validation set every epoch and log to TensorBoard
        if do_test and self.val_loader is not None:
            val_results = self._evaluate_and_log(split="val")
            curr_result = list(val_results.values())[0] if val_results else None
            
            # Handle best model saving if enabled
            if curr_result is not None and self.cfg.TEST.FINAL_MODEL == "best_val":
                is_best = curr_result > self.best_result
                if is_best:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        val_result=curr_result,
                        model_name="model-best.pth.tar"
                    )
                    print(f"New best validation result: {curr_result:.2f}% (saved as model-best.pth.tar)")
                elif self.best_result == -np.inf:
                    # First evaluation - always save as best model
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        val_result=curr_result,
                        model_name="model-best.pth.tar"
                    )
                    print(f"Initial validation result: {curr_result:.2f}% (saved as model-best.pth.tar)")
                else:
                    print(f"Validation result: {curr_result:.2f}% (best so far: {self.best_result:.2f}%)")
        elif do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            # Fallback: if no val_loader, use test set
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
                print(f"New best validation result: {curr_result:.2f}% (saved as model-best.pth.tar)")
            elif self.best_result == -np.inf:
                # First evaluation - always save as best model
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
                print(f"Initial validation result: {curr_result:.2f}% (saved as model-best.pth.tar)")
            else:
                print(f"Validation result: {curr_result:.2f}% (best so far: {self.best_result:.2f}%)")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def _evaluate_and_log(self, split=None):
        """Evaluate and log metrics without printing (for validation during training)."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            if split == "val":
                return None  # No validation set available
            split = "test"
            data_loader = self.test_loader

        # Track loss during validation
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            
            # Compute loss for this batch
            loss = F.cross_entropy(output, label)
            total_loss += loss.item()
            num_batches += 1

        results = self.evaluator.evaluate()
        
        # Add average loss to results
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            results["loss"] = avg_loss

        # Log all metrics to TensorBoard
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            if split == "val":
                print(f"Warning: Validation set requested but val_loader is None. Using test set instead.")
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # Don't log to TensorBoard here - this is for final evaluation only
        # Validation during training is logged by _evaluate_and_log()

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
