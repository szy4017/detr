from torch import nn
import torch
import torch.nn.functional as F

class MaskPredictor(nn.Module):
    """ predict mask score
    input:
        x: tensor([B, N, C])
        policy: tensor([B, N, 1])
    output:
        z: tensor([B, N, M])
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = (x[:, :, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class MaskLoss(nn.Module):
    """ calculate the loss of mask

    """
    def __init__(self, pruning_loc, token_ratio):
        super().__init__()
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

    def forward(self, pred_mask, pruning_index):
        if pruning_index not in self.pruning_loc:
            raise ValueError("this index is not in pruning")

        index = self.pruning_loc.index(pruning_index)
        pred_ratio = pred_mask.mean(1)
        mask_loss = torch.mean(((pred_ratio - self.token_ratio[index]) ** 2), dim=1, keepdim=True)

        return mask_loss


class Masking(nn.Module):
    """ update the decision mask
    input:
        x: tensor([B, N, C]), transformer feature map
        pre_mask: tensor([B, N, 1]), the previous mask
        pruning_layer_index: int, indicate the layer of mask
    output:
        post_mask: tensor([B, N, 1]), the post mask
    """
    def __init__(self, embed_dim=256, pruning_loc=[2, 4], token_ratio=[0.6, 0.3], num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.num_layers = num_layers
        self.loc, self.ratio_train, self.ratio_val = self.pruning_ratio_transform()
        self.mask_loss_cal = MaskLoss(self.loc, self.ratio_train)
        predictor_list = [MaskPredictor(self.embed_dim) for _ in range(len(self.loc))]
        self.mask_score_preict = nn.ModuleList(predictor_list)

    def pruning_ratio_transform(self):
        loc = []
        ratio_train = []
        ratio_val = []
        for l in range(0, self.num_layers):
            if l >= self.pruning_loc[0]:
                loc.append(l)

                if l in self.pruning_loc:
                    ratio_train.append(self.token_ratio[self.pruning_loc.index(l)])
                    ratio_val.append(self.token_ratio[self.pruning_loc.index(l)])
                else:
                    ratio_train.append(ratio_train[-1])
                    ratio_val.append(1.0)
        return loc, ratio_train, ratio_val

    def forward(self, x, pre_mask, pruning_index):
        if pruning_index not in self.loc:
            raise ValueError("this index is not in pruning")

        N, B, C = x.shape
        pred_mask_score = self.mask_score_preict[self.loc.index(pruning_index)](x.transpose(1, 0), pre_mask).reshape(B, -1, 2)
        if self.training:
            post_mask = F.gumbel_softmax(pred_mask_score, hard=True)[:, :, 0:1] * pre_mask
            mask_loss = self.mask_loss_cal(post_mask, pruning_index)
            return post_mask, mask_loss
        else:
            score = pred_mask_score[:, :, 0]
            keep_token_num = int(N * self.ratio_val[self.loc.index(pruning_index)])
            keep_policy = torch.argsort(score, dim=1, descending=True)[:, :keep_token_num]
            post_mask = batch_index_select(pre_mask, keep_policy)

            # save mask
            # import numpy as np
            # if self.pruning_loc.index(pruning_index) == 0:
            #     save_policy = keep_policy.cpu().numpy()
            #     np.save('policy_06.npy', save_policy)
            # if self.pruning_loc.index(pruning_index) == 3:
            #     save_policy = keep_policy.cpu().numpy()
            #     np.save('policy_03.npy', save_policy)

            # if self.pruning_loc.index(pruning_index) == 3:
            #     save_mask = pre_mask
            #     save_mask[:, keep_policy[0, :], :] = 0
            #     save_mask = save_mask.cpu().numpy()[0, :, 0]
            #     save_mask = save_mask.reshape((21, 42)).astype(int) * 255
            #     import cv2
            #     cv2.imwrite('mask.png', save_mask)

            return post_mask, keep_policy


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError



if __name__ == '__main__':
    # x = torch.rand((1, 20, 384))
    # pre_mask = torch.ones((1, 20, 1))
    # masking = Masking()
    # masking.eval()
    # post_mask, post_x = masking(x, pre_mask, pruning_index=3)
    masking = Masking(pruning_loc=[1, 3, 5], token_ratio=[0.1, 0.4, 0.7], num_layers=7)
    loc, ratio_train, ratio_val = masking.pruning_ratio_transform()
    print(loc)
    print(ratio_train)
    print(ratio_val)
    pass

