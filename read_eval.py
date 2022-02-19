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

def read_checkpont():
    checkpoint_path = '/home/szy/detr/results_pretrain_complete/checkpoint0199.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint.keys())

if __name__ == '__main__':
    #main()
    read_checkpont()