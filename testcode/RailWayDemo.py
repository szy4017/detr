import torch
import os
import cv2
import numpy as np
import torchvision.transforms.functional as F
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from util.misc import nested_tensor_from_tensor_list
from models.detr import PostProcess


class ToTensor(object):
    def __call__(self, img):
        return F.to_tensor(img)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image


def data_transform(frame):
    totensor = ToTensor()
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    H, W, C = frame.shape
    # cv2.imwrite('img.png', frame)
    resize_img = cv2.resize(frame, (W//2, H//2))
    img = totensor(resize_img)
    img = normalize(img)
    # cv2.imwrite('resize.png', resize_img)
    # print(frame.shape)
    return img


def build_dataset():
    cap = cv2.VideoCapture('../demo_1.mp4')

    f = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        f = f + 1

        if f % 10 == 0:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)


def show_results(frame, scores, labels, s_labels, boxes):
    plt.rcParams['figure.figsize'] = (15.0, 8.0)
    plt.figure()
    plt.imshow(frame)
    currentAxis = plt.gca()

    for score, label, s_label, box in zip(scores, labels, s_labels, boxes):
        if s_label == 1:
            rect_box = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        elif s_label == 2:
            rect_box = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect_box)

    plt.show()


def demo():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda')
    postprocess = PostProcess()
    # model = torch.load('../checkpoints/model.pth')
    model = torch.load('../checkpoints/model_v2.pth')
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture('../demo_2.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        H, W, C = frame.shape
        size = np.array([[H, W]])
        size = torch.from_numpy(size).to(device)
        if ret == False:
            continue

        img = data_transform(frame)
        x = nested_tensor_from_tensor_list([img])
        x = x.to(device)
        output = model(x)
        results = postprocess(output, size)
        scores = results[0]['scores'].cpu().numpy()
        labels = results[0]['labels'].cpu().numpy()
        s_labels = results[0]['s_labels'].cpu().numpy()
        boxes = results[0]['boxes'].cpu().numpy()
        save = np.where(scores > 0.5, True, False)

        scores = scores[save]
        labels = labels[save]
        s_labels = s_labels[save]
        boxes = boxes[save]

        show_results(frame, scores, labels, s_labels, boxes)

    pass


if __name__ == '__main__':
    demo()

    # build_dataset()