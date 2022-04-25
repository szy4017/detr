import torch
from torch import autograd
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def decoder_feature_test():
    # set arguments
    import argparse
    from pathlib import Path
    from main import get_args_parser
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # eval setting
    args.eval = True
    args.batch_size = 1
    args.dataset_file = 'intruscapes'
    # args.coco_path = '/home/szy/data/intruscapes' # for old server
    args.coco_path = '/data/szy4017/data/intruscapes'  # for new server
    args.output_dir = './results'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # model setting
    args.sta_query = True
    args.num_queries = 50
    args.ffn_model = 'old'
    args.aux_loss = True
    args.train_mode = 'finetune'
    args.resume = './results_pretrain_state_finetune_5/checkpoint.pth'

    args.distributed_mode = False

    device = torch.device(args.device)

    # build model
    from models import build_model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load('../results_pretrain_state_finetune_5/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # build dataset
    from torch.utils.data import DataLoader
    from datasets import build_dataset
    import util.misc as utils
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer'])

    # get grad
    for index, (samples, targets) in enumerate(data_loader_val):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        samples.tensors.requires_grad = True
        outputs = model(samples)



if __name__ == '__main__':
    # cnn_test()

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    decoder_feature_test()