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
    """
    用于将transformer部分的参数替换成经过预训练的参数
    """

    trans_checkpoint_path = '/home/szy/detr/checkpoints/detr-r50-e632da11.pth'
    #trans_checkpoint_path = '/home/szy/detr/results_pretrain_state_finetune/checkpoint0299.pth'
    trans_checkpoint = torch.load(trans_checkpoint_path, map_location='cpu')
    base_checkpoint_path = '/home/szy/detr/results_pretrain_state_finetune_1/checkpoint.pth'
    #base_checkpoint_path = '/home/szy/detr/base_checkpoint_3.pth'
    base_checkpoint = torch.load(base_checkpoint_path, map_location='cpu')

    print(trans_checkpoint.keys())
    print(trans_checkpoint['model'].keys())
    trans_model = trans_checkpoint['model']
    base_model = base_checkpoint['model']
    '''
    for key in trans_model.keys():
        if 'transformer' in key or 'backbone' in key:
            print(key)
            base_model.update({key: trans_model[key]})    
    '''

    for key in base_model.keys():
        if 'intru_state' in key or 'class' in key:
            print(key)
            trans_model.update({key: base_model[key]})

    print('\n')
    #print(base_model.keys())
    print(trans_model.keys())

    M = build_new_model()
    #M.load_state_dict(base_model)
    M.load_state_dict(trans_model)
    path = '/home/szy/detr/base_checkpoint_1_1.pth'
    torch.save({'model': M.state_dict()}, path)


def watch_checkpoint_classification():
    """
    查看checkpoint模型分类器结构
    """

    checkpoint_path = '/home/szy/detr/base_checkpoint_3.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model']

    print(checkpoint_path)
    for key in model.keys():
        if 'transformer' in key or 'backbone' in key:
            continue
        else:
            print(key)
            if 'class' in key:
                print(model[key].shape)

def read_eval():
    eval_path = '/home/szy/detr/results_pretrain_state_finetune/eval/latest.pth'
    eval = torch.load(eval_path, map_location='cpu')
    for key in eval.keys():
        print(key)


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
    log_path = '/home/szy/detr/results_pretrain_state_finetune/log.txt'
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

    plot(data_dict_4)


def plot(data_dict):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('State train and test error', fontsize=20)  # 标题，并设定字号大小
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
    plt.savefig('./4.png')  # 保存图片
    plt.show()  # 显示图像


def test_cross_entropy():
    import torch
    import torch.nn as nn
    x_input = torch.randn(3, 3)  # 随机生成输入
    print('x_input:\n', x_input)
    y_target = torch.tensor([1, 2, 2])  # 设置输出具体值 print('y_target\n',y_target)

    # 计算输入softmax，此时可以看到每一行加到一起结果都是1
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(x_input)
    print('soft_output:\n', soft_output)

    # 在softmax的基础上取log
    log_output = torch.log(soft_output)
    print('log_output:\n', log_output)

    # 对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func = nn.LogSoftmax(dim=1)
    logsoftmax_output = logsoftmax_func(x_input)
    print('logsoftmax_output:\n', logsoftmax_output)

    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func = nn.NLLLoss()
    nlloss_output = nllloss_func(logsoftmax_output, y_target)
    print('nlloss_output:\n', nlloss_output)

    # 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
    crossentropyloss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 100.0]))
    crossentropyloss_output = crossentropyloss(x_input, y_target)
    print('crossentropyloss_output:\n', crossentropyloss_output)



if __name__ == '__main__':
    #main()
    # read_checkpoint()
    #watch_checkpoint_classification()
    #build_new_model()
    #read_log()
    #read_eval()
    test_cross_entropy()