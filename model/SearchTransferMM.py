import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        # self.linear1 = nn.Linear(256*9, 128*9)
        # self.linear2 = nn.Linear(512*9, 128*9)

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv1, refsr_lv2, refsr_lv3, \
        ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1) # C 4H*4W 
        lrsr_lv2_unfold  = F.unfold(lrsr_lv2, kernel_size=(3, 3), padding=1) # 2C 2H*2W 
        lrsr_lv1_unfold  = F.unfold(lrsr_lv1, kernel_size=(3, 3), padding=1) # 4C H*W

        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1) # C 4H*4W    
        refsr_lv2_unfold = F.unfold(refsr_lv2, kernel_size=(3, 3), padding=1) # 2C 2H*2W
        # refsr_lv2_unfold = self.linear1(refsr_lv2_unfold)                     # C 2H*2W 
        refsr_lv1_unfold = F.unfold(refsr_lv1, kernel_size=(3, 3), padding=1) # 4C H*W 
        # refsr_lv1_unfold = self.linear1(refsr_lv1_unfold)                     # C H*W
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)
        refsr_lv2_unfold = refsr_lv2_unfold.permute(0, 2, 1)
        refsr_lv1_unfold = refsr_lv1_unfold.permute(0, 2, 1)
        
        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
        refsr_lv2_unfold = F.normalize(refsr_lv2_unfold, dim=2)
        refsr_lv1_unfold = F.normalize(refsr_lv1_unfold, dim=2)
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1)
        lrsr_lv2_unfold  = F.normalize(lrsr_lv2_unfold, dim=1)
        lrsr_lv1_unfold  = F.normalize(lrsr_lv1_unfold, dim=1)
        
        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, 4H*4W, 4H*4W]
        R_lv2 = torch.bmm(refsr_lv2_unfold, lrsr_lv2_unfold) #[N, 2H*2W, 2H*2W]
        R_lv1 = torch.bmm(refsr_lv1_unfold, lrsr_lv1_unfold) #[N, H*W, H*W]
        R_lv3 = R_lv3.unsqueeze(1)
        R_lv2 = R_lv2.unsqueeze(1)
        R_lv1 = R_lv1.unsqueeze(1)
        

        R = F.interpolate(R_lv2, scale_factor=4, mode='bicubic') +\
            F.interpolate(R_lv1, scale_factor=16, mode='bicubic') +\
            R_lv3
        R = R/3

        R = R[:,0,...]
         
        R_lv3_star, R_lv3_star_arg = torch.max(R, dim=1) #[N, 4H*4W]


        ### transfer
        # 重新排列组合
        
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1) # 4C H*W 
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2) # 2C 2H*2W 
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4) # C 4H*4W 

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S_3 = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3)) # 4H*4W


        return S_3, T_lv3, T_lv2, T_lv1