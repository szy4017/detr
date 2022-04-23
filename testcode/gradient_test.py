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

def simple_test():
    x = torch.tensor([1., 2.], requires_grad=True)
    a = torch.tensor([1., 1.], requires_grad=True)  # 对方程系数进行赋初值
    b = torch.tensor([2., 2.], requires_grad=True)
    c = torch.tensor([3., 3.], requires_grad=True)

    y = torch.sum(a ** 2 * x + b * x + c)

    # print('before:', a.grad, b.grad, c.grad)
    # grads = autograd.grad(y, [a, b, c])#利用函数自动求偏导
    # print('after a, b, c:', grads[0], grads[1], grads[2])

    print('before x:', x.grad)
    grads = autograd.grad(y, [x])  # 利用函数自动求偏导
    print('after x:', grads[0])


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)

    def forward(self, input):
        output = self.features(input)
        output = torch.mean(output)
        return output


def cnn_test():
    model = resnet18()
    input = cv2.imread('./test.png')
    input = input.transpose(2, 0, 1)
    input = np.expand_dims(input, axis=0)
    input = torch.from_numpy(input).float()
    input = F.normalize(input, p=2, dim=2)
    shape_aug = transforms.RandomResizedCrop(64, scale=(0.1, 1), ratio=(0.5, 2))
    input = shape_aug(input)
    input.requires_grad = True

    output = model(input)

    output.backward()
    print(input.grad.data.numpy())

    grad = autograd.grad(output, [input])
    input_grad = grad[0]
    input_grad = input_grad.mean(1)
    input_grad = F.normalize(input_grad, p=2, dim=2)
    input_grad = torch.sigmoid(input_grad)

    ## tensor可视化
    unloader = transforms.ToPILImage()
    image = input.clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('./example.jpg')

    image_grad = input_grad.clone()  # clone the tensor
    image_grad = image_grad.squeeze(0)  # remove the fake batch dimension
    image_grad = unloader(image_grad)
    image_grad.save('./example_grad.jpg')

    test_tensor = torch.ones((1, 1, 64, 64))
    image_test = test_tensor.squeeze(0)  # remove the fake batch dimension
    image_test = unloader(image_test)
    image_test.save('./example_test.jpg')


def detr_test():
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
    args.coco_path = '/home/szy/data/intruscapes' # for old server
    # args.coco_path = '/data/szy4017/data/intruscapes'  # for new server
    args.output_dir = './results'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # model setting
    args.sta_query = False
    args.num_queries = 50
    args.ffn_model = 'old'
    args.aux_loss = True
    args.train_mode = 'finetune'
    args.resume = './results_pretrain_state_finetune_9/checkpoint.pth'

    args.distributed_mode = False

    device = torch.device(args.device)

    # build model
    from models import build_model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load('../results_pretrain_state_finetune_9/checkpoint.pth', map_location='cpu')
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
    for samples, targets in data_loader_val:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        samples.tensors.requires_grad = True
        input_image = unnormalize(samples.tensors)
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        index_scores = np.where(results[0]['scores'].cpu().detach().numpy() > 0.5)
        index_labels = np.where(results[0]['labels'].cpu().detach().numpy() == 0)
        index_list = list()
        for s in index_scores[0]:
            for l in index_labels[0]:
                if s == l:
                    index_list.append(s)
        index_state = results[0]['s_labels'].cpu().detach().numpy()[index_list]

        for k, (i, s) in enumerate(zip(index_list, index_state)):
            get_output = outputs['pred_logits'][:, i, 0]
            get_output_state = outputs['pred_states'][:, i, s]
            get_box = results[0]['boxes'][i, :].cpu().detach().numpy()

            optimizer.zero_grad()
            get_output.backward(torch.ones_like(get_output), retain_graph=True)
            optimizer.zero_grad()
            grad = samples.tensors.grad.data.cpu()

            get_output_state.backward(torch.ones_like(get_output_state), retain_graph=True)
            grad_state = samples.tensors.grad.data.cpu()
            if k == 0:
                grad_new = grad
                grad_state_new = grad_state
            else:
                grad_new = grad - grad_old
                grad_state_new = grad_state - grad_state_old
            grad_old = grad
            grad_state_old = grad_state

            # visualize grad
            grad_new = TF.resize(grad_new, [1024, 2048])
            unloader = transforms.ToPILImage()
            grad_image = grad_new.clone()  # clone the tensor
            grad_image = grad_image.squeeze(0)  # remove the fake batch dimension
            grad_image = unloader(grad_image)
            grad_image = np.array(grad_image)
            grad_image = np.mean(grad_image, axis=-1)
            zero_image = np.zeros((1024, 2048, 3))
            zero_image[:, :, 0] = grad_image
            grad_image = zero_image

            plt.figure()
            currentAxis = plt.gca()
            rect = patches.Rectangle((get_box[0], get_box[1]), get_box[2]-get_box[0], get_box[3]-get_box[1], linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            plt.imshow(grad_image)
            plt.show()

            grad_state_new = TF.resize(grad_state_new, [1024, 2048])
            unloader = transforms.ToPILImage()
            grad_state_image = grad_state_new.clone()  # clone the tensor
            grad_state_image = grad_state_image.squeeze(0)  # remove the fake batch dimension
            grad_state_image = unloader(grad_state_image)
            grad_state_image = np.array(grad_state_image)
            grad_state_image = np.mean(grad_state_image, axis=-1)
            zero_image = np.zeros((1024, 2048, 3))
            zero_image[:, :, 2] = grad_state_image
            grad_state_image = zero_image

            plt.figure()
            currentAxis = plt.gca()
            rect = patches.Rectangle((get_box[0], get_box[1]), get_box[2]-get_box[0], get_box[3]-get_box[1], linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            plt.imshow(grad_state_image)
            plt.show()

            image = grad_image + grad_state_image
            plt.figure()
            currentAxis = plt.gca()
            rect = patches.Rectangle((get_box[0], get_box[1]), get_box[2]-get_box[0], get_box[3]-get_box[1], linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            plt.imshow(image)
            plt.show()

            grad_real_image = np.array(image)*0.5 + input_image
            plt.figure()
            currentAxis = plt.gca()
            rect = patches.Rectangle((get_box[0], get_box[1]), get_box[2]-get_box[0], get_box[3]-get_box[1], linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            plt.imshow(grad_real_image)
            plt.show()
            pass



def unnormalize(img):
    img = TF.resize(img, [1024, 2048])
    img = img.cpu().squeeze().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.485, 0.456, 0.406]
    img = img * std + mean     # unnormalize
    plt.figure()
    plt.imshow(img)
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

    return img


if __name__ == '__main__':
    # cnn_test()

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    detr_test()