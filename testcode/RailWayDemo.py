import torch
import os
import cv2
import numpy as np
import torchvision.transforms.functional as F

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


def demo():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda')
    postprocess = PostProcess()
    model = torch.load('../checkpoints/model.pth')
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture('../demo_1.mp4')
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
    pass


if __name__ == '__main__':
    demo()