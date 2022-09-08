from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
from glob import glob
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torchvision.utils as utils
import datetime
from torch.utils.tensorboard import SummaryWriter

today = datetime.date.today().strftime('%m%d')

log_testname = "./test_log/psnrssim %s.txt" % today

class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict, strict=False)

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('ref path:    %s' %(self.args.ref_path))
        psnr_sum = 0
        ssim_sum = 0

        # lr_list = sorted(glob(self.args.lr_path + "/*.jpg"))
        ref_list = sorted(glob(self.args.ref_path + "/*.png"))
        hr_list = sorted(glob(self.args.hr_path + "/*.png"))
        lr_list = sorted(glob(self.args.lr_path + "/*.png"))
        
        num = len(ref_list)
        for i in range (num):

            ### LR
            LR = imread(lr_list[i])
            ### HR 
            HR = imread(hr_list[i])
            h1, w1 = HR.shape[:2]
            h1, w1 = h1//4*4, w1//4*4
            HR = HR[:h1, :w1, :]
            LR_sr = np.array(Image.fromarray(LR).resize((w1, h1), Image.BICUBIC))
            
            ### Ref and Ref_sr
            Ref = imread(ref_list[i])
            h2, w2 = Ref.shape[:2]
            h2, w2 = h2//4*4, w2//4*4
            Ref = Ref[:h2, :w2, :]
            Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
            Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

            ### change type
            LR = LR.astype(np.float32)
            LR_sr = LR_sr.astype(np.float32)
            Ref = Ref.astype(np.float32)
            Ref_sr = Ref_sr.astype(np.float32)
            HR = HR.astype(np.float32)

            ### rgb range to [-1, 1]
            LR = LR / 127.5 - 1.
            LR_sr = LR_sr / 127.5 - 1.
            Ref = Ref / 127.5 - 1.
            Ref_sr = Ref_sr / 127.5 - 1.
            HR = HR/127.5 -1

            ### to tensor
            LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
            LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
            Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
            Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
            HR = torch.from_numpy(HR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

            self.model.eval()
            with torch.no_grad():
                sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
                sr_save = (sr+1.) * 127.5
                sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(ref_list[i]))
                imsave(save_path, sr_save)
                psnr_test, SSIM_test = calc_psnr_and_ssim(HR, sr)
                log_info = "图片%s--- PSNR:%4f, SSIM:%4f" %(os.path.basename(ref_list[i]),psnr_test,SSIM_test)
                print(log_info)
                psnr_sum += psnr_test
                ssim_sum += SSIM_test
                # 写入txt文件
                output_file = open(log_testname, 'a')
                output_file.write(log_info)
                output_file.write("\n")
                output_file.close()
                

        output_file = open(log_testname, 'a')
        output_file.write("PSNR_Avg:%4f,SSIM_Avg:%4f" %(psnr_sum/num,ssim_sum/num))
        output_file.write("\n")
        output_file.close()

        self.logger.info('Test over.')