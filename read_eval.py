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



if __name__ == '__main__':
    #main()
    read_checkpoint()
    #build_new_model()