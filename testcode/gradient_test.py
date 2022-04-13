import torch
from torch import autograd
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F
import cv2
import numpy as np

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
    input.requires_grad = True
    input = F.normalize(input, p=2, dim=2)
    shape_aug = transforms.RandomResizedCrop(64, scale=(0.1, 1), ratio=(0.5, 2))
    input = shape_aug(input)
    output = model(input)

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


if __name__ == '__main__':
    cnn_test()