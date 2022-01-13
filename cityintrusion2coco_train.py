import json
import os

def main():
    file_list = getFileList()
    json_path = "./Mycoco_train.json"
    with open(json_path, 'r') as json_file:
        annot = json.load(json_file)
    for id, file_name in enumerate(file_list):
        print(id)
        img_info = getImgInfo(file_name, id)
        annot['images'].append(img_info)

    f = open(json_path, 'w')
    annot_ = json.dumps(annot, indent=4)
    f.write(annot_)
    f.close()

def main_anns():
    file_list = getFileList()
    json_path = "./Mycoco_train.json"
    with open(json_path, 'r') as json_file:
        annot = json.load(json_file)
    for id, file_name in enumerate(file_list):
        print(id)
        anns_info = getAnns(file_name, id)
        annot['annotations'] = annot['annotations'] + anns_info

    f = open(json_path, 'w')
    annot_ = json.dumps(annot, indent=4)
    f.write(annot_)
    f.close()

def getFileList():
    file_path = '/home/szy/data/cityscape/gtIntrusionCityPersons/train'
    file_list = list()
    for i, j, k in os.walk(file_path):
        file_list = file_list + k
    print(file_list)
    print(len(file_list))
    return file_list

def getImgInfo(img_name, id):
    img_info = dict()
    img_info['id'] = id
    img_info['width'] = 2048
    img_info['height'] = 1024
    file_name = img_name.replace('gtIntrusionCityPersons.json', 'leftImg8bit.png')
    print(file_name)
    img_info['file_name'] = file_name
    city = file_name.split('_')[0]
    print(city)
    img_info['city'] = city
    print(img_info)
    return img_info

def getAnns(file_name, image_id):
    root_path = '/home/szy/data/cityscape/gtIntrusionCityPersons/train'
    ann_info_list = list()
    city_name = file_name.split('_')[0]
    file_path = os.path.join(root_path, city_name)
    file_path = os.path.join(file_path, file_name)
    print(file_path)
    with open(file_path, 'r') as json_file:
        annot = json.load(json_file)
    print(annot)
    for id, obj in enumerate(annot['objects']):
        if obj['label'] == 'pedestrian' or obj['label'] == 'rider':
            if obj['label'] == 'pedestrian':
                category_id = 1
                if obj.get('intru_label') is None:
                    state = -1
                else:
                    state = obj['intru_label']
            elif obj['label'] == 'rider':
                category_id = 2
                state = -1
            x, y, w, h = obj['bbox']
            area = float(w*h)
            ann_info = {
                'id': image_id*1000+id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [[]],
                'area': area,
                'bbox': [x, y, w, h],
                'iscrowd': 0,
                'state': state
            }
            ann_info_list.append(ann_info)
            print(ann_info)
    print(ann_info_list)
    return ann_info_list


if __name__ == '__main__':
    #main()
    #getFileList()
    #img_name = 'munster_000056_000019_gtIntrusionCityPersons.json'
    #getImgInfo(img_name, 1)
    #file_name = 'munster_000056_000019_gtIntrusionCityPersons.json'
    #getAnns(file_name, 1)
    main_anns()