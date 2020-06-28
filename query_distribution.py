import torch
import time
import torchvision

import numpy as np
import tqdm

import matplotlib.pyplot as plt

from datasets import build_dataset


def get_dataset(coco_path):
    """
    Gets the COCO dataset used for computing detections
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset

def compute_predictions(model, dataset, device):
    predictions = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            image, target = dataset[i]
            out = model([image.to(device)])
            res = out['pred_boxes'].cpu()
            predictions.append(res)

    preds = torch.cat(predictions, 0)
    return preds

PATH_TO_COCO = "/datasets01/COCO/022719"
dataset = get_dataset(PATH_TO_COCO)
device = torch.device('cuda')
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.to(device)
preds = compute_predictions(model, dataset, device)

s = (20, 4)
fig = plt.figure(figsize=s)
n = 10
for idx, query in enumerate(range(n * 2), 1):
    ax = fig.add_subplot(2, n, idx)
    p = preds[:, query]
    assert p.min() >= 0
    assert p.max() <= 1
    cx, cy, w, h = p.unbind(-1)
    area = (w * h) ** 0.5 * 10
    color = (w * h) ** 0.5
    color = torch.stack((w, 1 - color, h), 1)
    plt.scatter(cx, cy, c=color, s=area, alpha=0.75)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
fig.tight_layout()
plt.savefig('query_distribution.png', bbox_inches='tight')
