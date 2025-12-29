import os.path as osp

from dassl.utils import listdir_nohidden

from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class LengthDemo(DatasetBase):
    """Simple single-domain classification dataset with train/val splits."""

    dataset_dir = ""

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        train_dir = osp.join(root, cfg.DATASET.TRAIN_DIR)
        test_dir = osp.join(root, cfg.DATASET.TEST_DIR)

        train_x = self._read_split(train_dir)
        test = self._read_split(test_dir)

        super().__init__(train_x=train_x, test=test)

    def _read_split(self, split_dir):
        class_names = listdir_nohidden(split_dir)
        class_names.sort()

        items = []

        for label, class_name in enumerate(class_names):
            class_path = osp.join(split_dir, class_name)
            imnames = listdir_nohidden(class_path)

            for imname in imnames:
                impath = osp.join(class_path, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=class_name.lower()
                )
                items.append(item)

        return items

