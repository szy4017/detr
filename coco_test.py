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

    # 显示图片
    #show_image(train_set[1])
    #print(train_set[1])

    # 获取标注信息
    annFile_path = os.path.join(COCODIRS['annotations'], 'instances_val2017.json')
    coco = COCO(annFile_path)
    print(coco)

    # 获取类别标签信息
    print(coco.getCatIds())
    categories = coco.loadCats(coco.getCatIds())
    print(categories)
    names = [cat['name'] for cat in categories]
    print(names)
    catIds = coco.getCatIds(catNms=['person', 'bicycle', 'bus'])
    print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)
    print(imgIds)
    img_info = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
    img_info = img_info[0]
    print(img_info)
    annIds = coco.getAnnIds(catIds=catIds)
    anns_info = coco.loadAnns(annIds)
    #print(anns_info)

    # 在图片中显示标签
    imgPath = val_set[val_set.index(os.path.join(val_set_path, img_info['file_name']))]
    img = cv.imread(imgPath)
    plt.imshow(bgr2rgb(img))
    annIds = coco.getAnnIds(imgIds=img_info['id'])
    print(annIds)
    anns = coco.loadAnns(annIds)
    print(anns)
    print(anns[0]['bbox'])
    coco.showAnns(anns)
    coco.showBBox(anns)
    plt.show()

    # 获取关键点标注信息
    annKpFile_path = os.path.join(COCODIRS['annotations'], 'person_keypoints_val2017.json')
    coco_kps = COCO(annKpFile_path)
    annKpIds = coco_kps.getAnnIds(imgIds=img_info['id'])
    annKps = coco_kps.loadAnns(annKpIds)
    plt.imshow(bgr2rgb(img))
    coco_kps.showAnns(annKps)
    plt.show()

    # 获取字幕标注信息
    #annCpFile_path = os.path.join(COCODIRS['annotations'], 'captions_val2017.json')
    #coco_cps = COCO(annCpFile_path)
    #annCpIds = coco_cps.getAnnIds(imgIds=img_info['id'])
    #annCps = coco_cps.loadAnns(annCpIds)
    #print(annCps)
    #plt.imshow(bgr2rgb(img))
    #coco_kps.showAnns(annCps)
    #plt.show()

def getFileList():
    file_path = '/home/szy/data/cityscape/gtIntrusionCityPersons/val'
    file_list = list()
    for i, j, k in os.walk(file_path):
        file_list = file_list + k
    print(file_list)
    print(len(file_list))
    return file_list

def test_Mycoco():
    annFile_path = '/home/szy/detr/Mycoco.json'
    coco = COCO(annFile_path)
    print(coco)
    print(coco.getCatIds())
    categories = coco.loadCats(coco.getCatIds())
    print(categories)
    catIds = coco.getCatIds()[0]
    imgIds = coco.getImgIds(catIds=catIds)
    print(imgIds)
    img_info = coco.loadImgs(imgIds[0])
    print(img_info)
    img_info = img_info[0]
    annIds = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(annIds)
    print(anns)


def main_cityintrusion():
    # 构建验证集
    root_path = '/home/szy/data/cityscape/leftImg8bit/val'
    file_list = getFileList()
    val_set = list()
    for file_name in file_list:
        city_name = file_name.split('_')[0]
        file_path = os.path.join(root_path, city_name)
        file_path = os.path.join(file_path, file_name)
        val_set.append(file_path)
    print(val_set)
    print(len(val_set))

    # 获取标注信息
    annFile_path = './Mycoco_anns.json'
    coco = COCO(annFile_path)
    print(coco)

    # 获取类别标签信息
    print(coco.getCatIds())
    categories = coco.loadCats(coco.getCatIds())
    print(categories)
    names = [cat['name'] for cat in categories]
    print(names)
    catIds = coco.getCatIds(catNms='pedestrian')
    print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)
    print(imgIds)
    img_info = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
    img_info = img_info[0]
    print(img_info)

    root_path = '/home/szy/data/cityscape/leftImg8bit/val'
    file_name = img_info['file_name']
    city_name = file_name.split('_')[0]
    imgPath = os.path.join(root_path, city_name)
    imgPath = os.path.join(imgPath, file_name)
    print(imgPath)
    img = cv.imread(imgPath)
    plt.imshow(bgr2rgb(img))
    annIds = coco.getAnnIds(imgIds=img_info['id'])
    print(annIds)
    anns = coco.loadAnns(annIds)
    print(anns)
    print(anns[0]['bbox'])
    coco.showIntrusion(anns)

    plt.show()

def getFileList_train():
    file_path = '/home/szy/data/cityscape/gtIntrusionCityPersons/train'
    file_list = list()
    for i, j, k in os.walk(file_path):
        file_list = file_list + k
    print(file_list)
    print(len(file_list))
    return file_list


def main_cityintrusion_train():
    # 构建验证集
    root_path = '/home/szy/data/cityscape/leftImg8bit/train'
    file_list = getFileList_train()
    train_set = list()
    for file_name in file_list:
        city_name = file_name.split('_')[0]
        file_path = os.path.join(root_path, city_name)
        file_path = os.path.join(file_path, file_name)
        train_set.append(file_path)
    print(train_set)
    print(len(train_set))

    # 获取标注信息
    annFile_path = './Mycoco_train.json'
    coco = COCO(annFile_path)
    print(coco)

    # 获取类别标签信息
    print(coco.getCatIds())
    categories = coco.loadCats(coco.getCatIds())
    print(categories)
    names = [cat['name'] for cat in categories]
    print(names)
    catIds = coco.getCatIds(catNms='pedestrian')
    print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)
    print(imgIds)
    img_info = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
    img_info = img_info[0]
    print(img_info)

    root_path = '/home/szy/data/cityscape/leftImg8bit/train'
    file_name = img_info['file_name']
    city_name = file_name.split('_')[0]
    imgPath = os.path.join(root_path, city_name)
    imgPath = os.path.join(imgPath, file_name)
    print(imgPath)
    img = cv.imread(imgPath)
    plt.imshow(bgr2rgb(img))
    annIds = coco.getAnnIds(imgIds=img_info['id'])
    print(annIds)
    anns = coco.loadAnns(annIds)
    print(anns)
    print(anns[0]['bbox'])
    coco.showBBox(anns)
    #coco.showIntrusion(anns)

    plt.show()

if __name__ == '__main__':
    #test_Mycoco()
    #main()
    #main_cityintrusion()
    main_cityintrusion_train()