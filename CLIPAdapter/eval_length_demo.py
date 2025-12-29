import argparse, os.path as osp, sys, torch

# Ensure project and Dassl repo are on sys.path
BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
DASSL_DIR = osp.join(BASE_DIR, "Dassl.pytorch-master")
for p in (BASE_DIR, DASSL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import CLIPAdapter.CLIPAdapter  # noqa: E402,F401 ensures trainer is registered
import importlib

_cfg_mod = importlib.import_module("dassl.config")
_engine_mod = importlib.import_module("dassl.engine")
_utils_mod = importlib.import_module("dassl.utils")
get_cfg_default, clean_cfg = _cfg_mod.get_cfg_default, _cfg_mod.clean_cfg
build_trainer = _engine_mod.build_trainer
setup_logger = _utils_mod.setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", type=str, default="../CLIPAdapter/config_length_demo.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file (e.g., output/clip_adapter/model.pth.tar-XX)")
    p.add_argument("--test-dir", type=str, default="", help="Override TEST_DIR (relative or absolute folder)")
    p.add_argument("--root", type=str, default="", help="Override DATASET.ROOT")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg_default()
    cfg.merge_from_file(args.config_file)
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.test_dir:
        if osp.isabs(args.test_dir):
            cfg.DATASET.ROOT = osp.dirname(args.test_dir)
            cfg.DATASET.TEST_DIR = osp.basename(args.test_dir)
        else:
            cfg.DATASET.TEST_DIR = args.test_dir
    clean_cfg(cfg, cfg.TRAINER.NAME)
    cfg.freeze()
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    trainer.load_model(osp.dirname(args.checkpoint), epoch=None)
    trainer.test()


if __name__ == "__main__":
    main()

