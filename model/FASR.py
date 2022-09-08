from model import CHPF, LTE, SearchTransferMM, SearchTransferSM

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)
                     
class FASR(nn.Module):
    def __init__(self, args):
        super(FASR, self).__init__()
        self.args = args
        self.num_res_blocks = [16,16,8,4]
        self.CHPF = CHPF.CHPF(num_res_blocks=self.num_res_blocks, n_feats=64, 
            res_scale=1)

        # MM
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_ref = LTE.LTE(requires_grad=True, out_switch=True)
        self.LTE_lrsr = LTE.LTE(requires_grad=True, out_switch=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.conv1 = conv3x3(512, 256)
        self.conv2 = conv3x3(256, 128)
        self.conv3 = conv3x3(128, 64)

        # SM
        self.LTE_SM      = LTE.LTE(requires_grad=True)
        self.LTE_ref_SM = LTE.LTE(requires_grad=True)
        self.LTE_lrsr_SM = LTE.LTE(requires_grad=True)
        self.LTE_copy_SM = LTE.LTE(requires_grad=False)

        self.SearchTransfer = SearchTransferMM.SearchTransfer()
        self.SearchTransfer_SM = SearchTransferSM.SearchTransfer()

    def forward(self, lr, lrsr, ref, refsr, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy_SM((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        # M-M
        lrsr_lv3, lrsr_lv2, lrsr_lv1  = self.LTE_lrsr((lrsr.detach() + 1.) / 2.)
        
        refsr_lv3, refsr_lv2, refsr_lv1 = self.LTE_ref((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        S3, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv1, refsr_lv2, \
                            refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        
        # S-M
        _, _, lrsr_lv3_SM  = self.LTE_lrsr_SM((lrsr.detach() + 1.) / 2.)

        refsr_lv1_SM, refsr_lv2_SM, refsr_lv3_SM = self.LTE_ref_SM((refsr.detach() + 1.) / 2.)

        ref_lv1_SM, ref_lv2_SM, ref_lv3_SM = self.LTE_SM((ref.detach() + 1.) / 2.)

        S1_SM, S2_SM, S3_SM, T_lv3_SM, T_lv2_SM, T_lv1_SM = self.SearchTransfer_SM(lrsr_lv3_SM, refsr_lv1_SM, refsr_lv2_SM, \
                            refsr_lv3_SM, ref_lv1_SM, ref_lv2_SM, ref_lv3_SM)

        S1 = (S1_SM + S3) / 2
        S2 = (S1_SM + S3) / 2
        S3 = (S1_SM + S3) / 2
        
        T_lv3 = torch.cat((T_lv3, T_lv3_SM), dim = 1)
        T_lv2 = torch.cat((T_lv2, T_lv2_SM), dim = 1)
        T_lv1 = torch.cat((T_lv1, T_lv1_SM), dim = 1)
        
        T_lv3 = self.conv1(T_lv3)
        T_lv2 = self.conv2(T_lv2)
        T_lv1 = self.conv3(T_lv1)



        sr = self.CHPF(lr, S1, S2, S3, T_lv3, T_lv2, T_lv1)


        return sr, S3 ,T_lv3, T_lv2, T_lv1 
