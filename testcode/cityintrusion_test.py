import sys, os
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pycocotools.coco import COCO
import json

def bgr2rgb(img):
    # 用cv自带的分割和合并函数
    B, G, R = cv.split(img)
    return cv.merge([R, G, B])

def show_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    img = bgr2rgb(img)
    plt.imshow(img)
    plt.show()

def main():
    # 构建验证集
    CityintrusionPATH = '/home/szy/data/intruscapes'
    CityintrusionDIRS = os.listdir(CityintrusionPATH)

    CityintrusionDIRS = {  # 改用字典比较好访问
        d: os.path.join(CityintrusionPATH, d) for d in CityintrusionDIRS
    }
    print(CityintrusionDIRS)

    val_set_path = os.path.join(CityintrusionDIRS['images'], 'val')
    val_set = [os.path.join(val_set_path, p) for p in os.listdir(val_set_path)]
    print(val_set[:5])

    # 获取标注信息
    annFile_path = os.path.join(CityintrusionDIRS['annotations'], 'instances_val.json')
    cityintru = COCO(annFile_path)
    print(cityintru)

    # 获取类别标签信息
    print(cityintru.getCatIds())
    categories = cityintru.loadCats(cityintru.getCatIds())
    print(categories)
    names = [cat['name'] for cat in categories]
    print(names)
    catIds = cityintru.getCatIds(catNms='pedestrian')
    print(catIds)
    imgIds = cityintru.getImgIds(catIds=catIds)
    print(imgIds)
    img_info = cityintru.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
    img_info = img_info[0]
    print(img_info)

    # 在图片中显示标签
    imgPath = val_set[val_set.index(os.path.join(val_set_path, img_info['file_name']))]
    print(imgPath)
    img = cv.imread(imgPath)
    plt.imshow(bgr2rgb(img))
    annIds = cityintru.getAnnIds(imgIds=img_info['id'])
    print(annIds)
    anns = cityintru.loadAnns(annIds)
    print(anns)
    print(anns[0]['bbox'])
    cityintru.showIntrusion(anns)
    plt.show()


def main_train():
    # 构建验证集
    CityintrusionPATH = '/home/szy/data/intruscapes'
    CityintrusionDIRS = os.listdir(CityintrusionPATH)

    CityintrusionDIRS = {  # 改用字典比较好访问
        d: os.path.join(CityintrusionPATH, d) for d in CityintrusionDIRS
    }
    print(CityintrusionDIRS)

    train_set_path = os.path.join(CityintrusionDIRS['images'], 'train')
    train_set = [os.path.join(train_set_path, p) for p in os.listdir(train_set_path)]
    print(train_set[:5])

    # 获取标注信息
    annFile_path = os.path.join(CityintrusionDIRS['annotations'], 'instances_train_c.json')
    cityintru = COCO(annFile_path)
    print(cityintru)

    # 获取类别标签信息
    print(cityintru.getCatIds())
    categories = cityintru.loadCats(cityintru.getCatIds())
    print(categories)
    names = [cat['name'] for cat in categories]
    print(names)
    catIds = cityintru.getCatIds(catNms='pedestrian')
    print(catIds)
    imgIds = cityintru.getImgIds(catIds=catIds)
    print(imgIds)
    print(len(cityintru.imgs))
    img_info = cityintru.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
    img_info = img_info[0]
    print(img_info)

    # 在图片中显示标签
    imgPath = train_set[train_set.index(os.path.join(train_set_path, img_info['file_name']))]
    print(imgPath)
    img = cv.imread(imgPath)
    plt.imshow(bgr2rgb(img))
    annIds = cityintru.getAnnIds(imgIds=img_info['id'])
    print(annIds)
    anns = cityintru.loadAnns(annIds)
    print(anns)
    print(anns[0]['bbox'])
    cityintru.showIntrusion(anns)
    plt.show()

if __name__ == '__main__':
    #main()
    main_train()