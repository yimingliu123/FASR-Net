import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        self.linear1 = nn.Linear(128*9, 256*9)
        self.linear2 = nn.Linear(64*9, 256*9)

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

    def forward(self, lrsr_lv3, refsr_lv1, refsr_lv2, refsr_lv3, \
        ref_lv1, ref_lv2, ref_lv3):
        ### search
        # 低分辨率bicubic上采样后的特征patch化
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1) # 4C*k*k H*W    256*9 
        # lrsr_lv2_unfold  = F.unfold(lrsr_lv2, kernel_size=(3, 3), padding=1) # 2C 2H*2W 
        # lrsr_lv1_unfold  = F.unfold(lrsr_lv1, kernel_size=(3, 3), padding=1) # 4C H*W

        # refbicubic下上采样后的特征patch化 
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1) # 4C*k*k H*W     256*9   
        refsr_lv2_unfold = F.unfold(refsr_lv2, kernel_size=(3, 3), padding=1) # 2C*k*k 2H*2W   128*9
        refsr_lv1_unfold = F.unfold(refsr_lv1, kernel_size=(3, 3), padding=1) # C*K*K 4H*4W                    
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)                 #   H*W 4C*k*k 
        refsr_lv2_unfold = refsr_lv2_unfold.permute(0, 2, 1)                 #  2H*2W 2C*k*k
        refsr_lv2_unfold = self.linear1(refsr_lv2_unfold)                   
        refsr_lv1_unfold = refsr_lv1_unfold.permute(0, 2, 1)                 #  4H*4W  C*k*k
        refsr_lv1_unfold = self.linear2(refsr_lv1_unfold) 
        
        # 归一化
        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
        refsr_lv2_unfold = F.normalize(refsr_lv2_unfold, dim=2)
        refsr_lv1_unfold = F.normalize(refsr_lv1_unfold, dim=2)
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1)
        # lrsr_lv2_unfold  = F.normalize(lrsr_lv2_unfold, dim=1)
        # lrsr_lv1_unfold  = F.normalize(lrsr_lv1_unfold, dim=1)
        
        # 相关计算，最大值索引
        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N,  H*W, H*W]
        R_lv2 = torch.bmm(refsr_lv2_unfold, lrsr_lv3_unfold) #[N, 2H*2W, H*W]
        R_lv1 = torch.bmm(refsr_lv1_unfold, lrsr_lv3_unfold) #[N, 4H*4W, H*W]

        # R_lv3 = R_lv3.unsqueeze(1)
        # R_lv2 = R_lv2.unsqueeze(1)
        # R_lv1 = R_lv1.unsqueeze(1)

        # R_2 = F.interpolate(R_lv2, scale_factor=(0.25, 1), mode = "bicubic")
        # R_1 = F.interpolate(R_lv1, scale_factor=(1/16, 1), mode = "bicubic")
        # R = (R_lv3+R_2+R_1)/3
        # R = R[:,0,...]
         
        # R_lv_star, R_lv_star_arg = torch.max(R, dim=1) #[N,  H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N,  H*W]
        R_lv2_star, R_lv2_star_arg = torch.max(R_lv2, dim=1) #[N,  H*W]
        R_lv1_star, R_lv1_star_arg = torch.max(R_lv1, dim=1) #[N,  H*W]
        # R_lv2_star, R_lv2_star_arg = torch.max(R_lv2, dim=1) #[N,  H*W]
        # R_lv1_star, R_lv1_star_arg = torch.max(R_lv1, dim=1) #[N,  H*W]
        # print(R_lv3_star.shape)
        # print(R_lv2_star.shape)
        # print(R_lv1_star.shape)
        ### transfer
        # 重新排列组合
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1) # 4C H*W 
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2) # 2C 2H*2W 
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4) # C 4H*4W 
        # print(ref_lv2_unfold.shape)
        # print(R_lv2_star_arg.shape)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
        # print(T_lv3.shape)
        # print(T_lv2.shape)
        # print(T_lv1.shape)

        S_3 = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3)) # 4H*4W
        S_2 = R_lv2_star.view(R_lv2_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3)) # 4H*4W
        S_1 = R_lv1_star.view(R_lv1_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3)) # 4H*4W


        return S_1, S_2, S_3, T_lv3, T_lv2, T_lv1