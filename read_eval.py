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


if __name__ == '__main__':
    main()