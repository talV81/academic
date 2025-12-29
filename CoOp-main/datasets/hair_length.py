import os
import pickle
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


@DATASET_REGISTRY.register()
class HairLength(DatasetBase):

    dataset_dir = "hair_length"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train = self.read_data(os.path.join(self.dataset_dir, "train"))
        val = self.read_data(os.path.join(self.dataset_dir, "val"))
        test = self.read_data(os.path.join(self.dataset_dir, "test"))

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_dir):
        items = []
        class_names = sorted(listdir_nohidden(split_dir))
        
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            imnames = listdir_nohidden(class_dir)
            
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                items.append(item)
        
        return items

