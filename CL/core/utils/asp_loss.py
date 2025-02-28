import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
__all__ = ['AdaptiveSupervisedPatchNCELoss']

class AdaptiveSupervisedPatchNCELoss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.total_epochs = opt.n_epochs + opt.n_epochs_decay

    def forward(self, feat_q, feat_k, current_epoch=-1): # q：target，k：source
        # 重新调整输入维度
        # feat_q: anchor，HE
        # feat_k：anchor，IHC
        # print('feat_q',feat_q.shape) # torch.Size([11, 2048, 512])
        if len(feat_q.shape) == 3:  # 如果输入是 [batch, num_patches, dim]
            batch_size = feat_q.shape[0]
            num_patches = feat_q.shape[1]
            dim = feat_q.shape[2] # 512
            # 重塑为 [batch*num_patches, dim]
            feat_q = feat_q.reshape(-1, dim)
            feat_k = feat_k.reshape(-1, dim)
            print('feat_q:',feat_q.shape)

        else:  # 原始情况 [num_patches, dim]
            num_patches = feat_q.shape[0]
            dim = feat_q.shape[1]
            batch_size = 1

        feat_k = feat_k.detach()
        print('feat_k:',feat_k.shape)

        # pos logit，正样本相似度得分
        l_pos = torch.bmm(
            feat_q.view(num_patches * batch_size, 1, -1), # 计算q和k之间的相似度
            feat_k.view(num_patches * batch_size, -1, 1))
        l_pos = l_pos.view(num_patches * batch_size, 1)
        print('l_pos:',l_pos.shape)
        # print('nce_includes_all_negatives_from_minibatch:',self.opt.nce_includes_all_negatives_from_minibatch)
        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch: # True
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size * batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        print('feat_q:',feat_q.shape)

        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        print('l_neg_curbatch:',l_neg_curbatch.shape)
        
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        if self.opt.asp_loss_mode == 'none':
            return loss

        scheduler, lookup = self.opt.asp_loss_mode.split('_')[:2]
        # Compute scheduling
        t = (current_epoch - 1) / self.total_epochs
        if scheduler == 'sigmoid':
            p = 1 / (1 + np.exp((t - 0.5) * 10))
        elif scheduler == 'linear':
            p = 1 - t
        elif scheduler == 'lambda':
            k = 1 - self.opt.n_epochs_decay / self.total_epochs
            m = 1 / (1 - k)
            p = m - m * t if t >= k else 1.0
        elif scheduler == 'zero':
            p = 1.0
        else:
            raise ValueError(f"Unrecognized scheduler: {scheduler}")
        # Weight lookups
        w0 = 1.0
        x = l_pos.squeeze().detach()
        
        # 添加检查点1
        # print(f"l_pos range: min={x.min().item()}, max={x.max().item()}")
        
        if lookup == 'top':
            x = torch.where(x > 0.0, x, torch.zeros_like(x))
            w1 = torch.sqrt(1 - (x - 1) ** 2)
        elif lookup == 'linear':
            w1 = torch.relu(x)
        elif lookup == 'bell':
            sigma, mu, sc = 1, 0, 4
            w1 = 1 / (sigma * np.sqrt(2 * torch.pi)) * torch.exp(-((x - 0.5) * sc - mu) ** 2 / (2 * sigma ** 2))
        elif lookup == 'uniform':
            w1 = torch.ones_like(x)
        else:
            raise ValueError(f"Unrecognized lookup: {lookup}")
            
        # 添加检查点2
        # print(f"w1 range: min={w1.min().item()}, max={w1.max().item()}") # w1 range: min=nan, max=nan
        
        # Apply weights with schedule
        w = p * w0 + (1 - p) * w1
        # Normalize
        w = w / w.sum() * len(w)
        # 添加检查点4
        # print(f"Post-norm w range: min={w.min().item()}, max={w.max().item()}")
        loss = loss * w
        
        # 添加检查点5
        # print(f"Weighted loss range: min={loss.min().item()}, max={loss.max().item()}")
        
        loss = loss.mean()
        
        # 最终检查点
        # if torch.isnan(loss):
        #     print("Final loss is NaN!")
            
        return loss
