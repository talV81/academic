import cv2
import numpy as np
import os


dataset_path = '/home/tal/Private/KITA/cracks/crack_segmentation_dataset'
partitions_folders = ['train', 'test']
output_folder = 'labels'

for partition in partitions_folders:
    partition_path = os.path.join(dataset_path, partition)
    input_masks_path = os.path.join(partition_path, 'masks')
    output_path = os.path.join(partition_path, 'labels')
    os.makedirs(output_path, exist_ok=True)

    for img_name in os.listdir(input_masks_path):
        img_name_png = img_name.replace('.jpg', '.png')
        mask = cv2.imread(os.path.join(input_masks_path, img_name), 0)
        labels_mask = np.round(mask / 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, img_name_png), labels_mask)

