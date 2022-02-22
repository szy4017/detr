import torch

def main():
    pthfile = '/home/szy/detr/results/eval/000.pth'
    model = torch.load(pthfile, map_location=torch.device('cpu'))

    print("type:")
    print(type(model))

    print('keys:')
    for k in model.keys():
        print(k)

    print(model['params'])
    print(model['counts'])
    print(model['date'])
    precision = model['precision']
    print(precision)

    '''
    print('values:')
    for k in model:
        print(k, model[k])    
    '''

def read_checkpoint():
    test_checkpoint_path = '/home/szy/detr/test_checkpoint.pth'
    test_checkpoint = torch.load(test_checkpoint_path, map_location='cpu')
    checkpoint_path = '/home/szy/detr/results_pretrain_complete/checkpoint0299.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(test_checkpoint.keys())
    print(test_checkpoint['model'].keys())
    test_model = test_checkpoint['model']
    model = checkpoint['model']
    for key in test_model.keys():
        if 'intru' in key:
            print(key)
            model.update({key: test_model[key]})

    print('\n')
    print(model.keys())

    M = build_new_model()
    M.load_state_dict(model)
    path = '/home/szy/detr/intru_checkpoint.pth'
    torch.save({'model': M.state_dict()}, path)


def build_new_model():
    import os
    from main import get_args_parser
    from pathlib import Path
    import argparse
    from models import build_model

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # for evaluation
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = './results'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.batch_size = 2
    args.no_aux_loss = True
    args.eval = True
    #args.resume = '/home/szy/detr/checkpoints/detr-r50-e632da11.pth'
    #args.resume = '/home/szy/detr/results/checkpoint0299.pth'
    args.resume = '/home/szy/detr/results_pretrain_complete/checkpoint0199.pth'
    #args.dataset_file = 'coco'
    args.dataset_file = 'intruscapes'
    #args.coco_path = '/home/szy/data/coco'
    args.coco_path = '/home/szy/data/intruscapes'

    model, criterion, postprocessors = build_model(args)  # 构建模型
    return model


def read_log():
    import json
    log_path = '/home/szy/detr/results_pretrain_state/log.txt'
    log_file = open(log_path, 'r')
    data_dict_1 = {'train_class_error': [], 'train_state_error_unscaled': [], 'epoch': []}
    data_dict_2 = {'train_loss_ce': [], 'train_loss_se_unscaled': [], 'epoch': []}
    data_dict_3 = {'train_class_error': [], 'test_class_error': [], 'epoch': []}
    data_dict_4 = {'train_state_error_unscaled': [], 'test_state_error_unscaled': [], 'epoch': []}
    for line in log_file:
        # json.loads()可以字典格式的str转换成dict
        log_dic = json.loads(line)
        for key in log_dic.keys():
            if key == 'epoch':
                data_dict_1[key].append(log_dic[key])
                data_dict_2[key].append(log_dic[key])
                data_dict_3[key].append(log_dic[key])
                data_dict_4[key].append(log_dic[key])
            if key == 'train_class_error':
                data_dict_1[key].append(log_dic[key])
                data_dict_3[key].append(log_dic[key])
            if key == 'train_state_error_unscaled':
                data_dict_1[key].append(log_dic[key])
                data_dict_4[key].append(log_dic[key])
            if key == 'train_loss_ce':
                data_dict_2[key].append(log_dic[key])
            if key == 'train_loss_se_unscaled':
                data_dict_2[key].append(log_dic[key])
            if key == 'test_state_error_unscaled':
                data_dict_4[key].append(log_dic[key])
            if key == 'test_class_error':
                data_dict_3[key].append(log_dic[key])

    plot(data_dict_1)


def plot(data_dict):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Class and State error', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel(u'epoch', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'error', fontsize=14)  # 设置y轴，并设定字号大小

    y = {}
    for key, value in data_dict.items():
        if 'epoch' in key:
            x = data_dict[key]
        else:
            y[key] = data_dict[key]

    color = ["deeppink", "darkblue"]
    i = 0
    for key, value in y.items():
        # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
        plt.plot(x, value, color=color[i], linewidth=2, linestyle=':', label=key, marker='o')
        i = i+1

    plt.legend(loc=2)  # 图例展示位置，数字代表第几象限
    plt.savefig('./1.png')  # 保存图片
    plt.show()  # 显示图像


if __name__ == '__main__':
    #main()
    #read_checkpoint()
    #build_new_model()
    read_log()