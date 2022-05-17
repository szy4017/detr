import torch

def read_checkpoint():
    base_checkpoint_path = '/data/szy4017/code/detr/checkpoints/detr-r50-e632da11.pth'
    old_checkpoint_path = '/data/szy4017/code/detr/results_pretrain_state_finetune_1/checkpoint.pth'
    old_checkpoint_path_1 = '/data/szy4017/code/detr/results_pretrain_state_finetune_1/checkpoint0299.pth'
    old_checkpoint_path_2 = '/data/szy4017/code/detr/results_pretrain_state_finetune_1/checkpoint0399.pth'
    new_checkpoint_path = '/data/szy4017/code/detr/results_repeat_staquery_inbackbone_4/checkpoint.pth'
    new_checkpoint_path_1 = '/data/szy4017/code/detr/results_repeat_staquery_inbackbone_4/checkpoint0099.pth'
    new_checkpoint_path_2 = '/data/szy4017/code/detr/results_repeat_staquery_inbackbone_4/checkpoint0199.pth'

    base_model = torch.load(base_checkpoint_path, map_location=torch.device('cpu'))['model']
    old_model = torch.load(old_checkpoint_path, map_location=torch.device('cpu'))['model']
    old_model_1 = torch.load(old_checkpoint_path_1, map_location=torch.device('cpu'))['model']
    old_model_2 = torch.load(old_checkpoint_path_2, map_location=torch.device('cpu'))['model']
    new_model = torch.load(new_checkpoint_path, map_location=torch.device('cpu'))['model']
    new_model_1 = torch.load(new_checkpoint_path_1, map_location=torch.device('cpu'))['model']
    new_model_2 = torch.load(new_checkpoint_path_2, map_location=torch.device('cpu'))['model']

    # for k, v in base_model.items():
    #     if not (k in old_model.keys()):
    #         print('differences key between base and old: ', k)
    #     else:
    #         if not (v.shape == old_model[k].shape):
    #             print('differences value between base and old: ', k, ' ', v)
    #
    #     if not (k in new_model.keys()):
    #         print('differences key between base and new: ', k)
    #     else:
    #         if not (v.shape == new_model[k].shape):
    #             print('differences value between base and new: ', k, ' ', v)
    #     print('-----------------')


    # for k, v in old_model.items():
    #     if not (k in base_model.keys()):
    #         print('differences key between old and new: ', k)
    #         print(v)
    #         print(v.shape)
    #     else:
    #         if not (v.shape == base_model[k].shape):
    #             print('differences value between base and old: ', k, ' ', v)
    #
    #     print('-----------------')

    for (k1, v1), (k2, v2) in zip(new_model_1.items(), new_model_2.items()):
        if 'query' in k1:
            print(k1)
            print(v1)
            print(v1.shape)
        else:
            print('**************')

        if 'query' in k2:
            print(k2)
            print(v2)
            print(v2.shape)
        else:
            print('--------------')


if __name__ == '__main__':
    read_checkpoint()