import sys, os
from matplotlib import pyplot as plt
import cv2 as cv
img = cv.imread(train_set[1], cv.IMREAD_UNCHANGED) # 随便加载数据集的一张图
# 注意cv的通道顺序和matplotlib的不一样，下面是未转换成RGB顺序

def show_image(image_path):


def main():
    COCOPATH = '/home/szy/data/coco'
    COCODIRS = os.listdir(COCOPATH)

    COCODIRS = {  # 改用字典比较好访问
        d: os.path.join(COCOPATH, d) for d in COCODIRS
    }
    print(COCODIRS)

    # 构建训练集
    train_set_path = os.path.join(COCODIRS['images'], 'train2017')
    train_set = [os.path.join(train_set_path, p) for p in os.listdir(train_set_path)]
    print(train_set[:5])  # 查看部分训练集（图片）

    # 构建验证集
    val_set_path = os.path.join(COCODIRS['images'], 'val2017')
    val_set = [os.path.join(val_set_path, p) for p in os.listdir(val_set_path)]
    print(val_set[:5])


if __name__ == '__main__':
    main()