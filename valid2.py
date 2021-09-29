import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls
import utils.metric as mc
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import model.siameseNet.deeplab_v2 as models

### options = ['TSUNAMI','GSV','CMU','CD2014']
datasets = 'CD2014'
if datasets == 'TSUNAMI':
    import cfgs.TSUNAMIconfig as cfg
if datasets == 'GSV':
    import cfgs.GSVconfig as cfg
if datasets == 'CMU':
    import cfgs.CMUconfig as cfg
if datasets == 'CD2014':
    import cfgs.CD2014config as cfg


def data_transform(img1,img2,lbl):
    img1 = img1[:, :, ::-1]  # RGB -> BGR
    img1 = img1.astype(np.float64)
    img1 -= cfg.T0_MEAN_VALUE
    img1 = img1.transpose(2, 0, 1)
    img1 = torch.from_numpy(img1).float()
    img2 = img2[:, :, ::-1]  # RGB -> BGR
    img2 = img2.astype(np.float64)
    img2 -= cfg.T1_MEAN_VALUE
    img2 = img2.transpose(2, 0, 1)
    img2 = torch.from_numpy(img2).float()
    lbl = torch.from_numpy(lbl).long()
    return img1,img2,lbl

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance


testCase1_01 = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/t0/60.jpg'
testCase1_02 = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/t1/60.jpg'
testCase1_bg = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/bg/1.png'


img1 = Image.open(testCase1_01)
img2 = Image.open(testCase1_02)
imgbg = Image.open(testCase1_bg)

img1 = img1.resize((1024,768),Image.ANTIALIAS)
img2 = img2.resize((1024,768),Image.ANTIALIAS)
imgbg = imgbg.resize((1024,768),Image.ANTIALIAS)


height,width,_ = np.array(img1,dtype= np.uint8).shape
img1 = np.array(img1,dtype= np.uint8)
img2 = np.array(img2,dtype= np.uint8)
#label = np.zeros((height,width,3),dtype=np.uint8)
label = np.array(imgbg,dtype=np.uint8)

img1, img2, label = data_transform(img1, img2, label)
inputs1,input2, targets = img1, img2, label
model = models.SiameseNet(norm_flag='l2')


#pretrain_deeplab_path = "/home/m0729013/SceneChangeDet/code/pretrain/deeplab_v2_voc12.pth"
#deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
#model.init_parameters_from_deeplab(deeplab_pretrain_model)
TRAINED_BEST_PERFORMANCE_CKPT = "/home/m0729013/SceneChangeDet/code/pretrain/model95.pth"
checkpoint = torch.load(TRAINED_BEST_PERFORMANCE_CKPT)#The main different parts with https://github.com/gmayday1997/SceneChangeDet/issues/19
model.load_state_dict(checkpoint['state_dict'])#The main different parts with https://github.com/gmayday1997/SceneChangeDet/issues/19

model = model.cuda()
model.eval()

cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
metric_for_conditions = util.init_metric_for_class_for_cmu(1)

inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0), volatile=True),Variable(input2.unsqueeze(0),volatile=True) ,Variable(targets)
out_conv5,out_fc,out_embedding = model(inputs1,inputs2)
out_conv5_t0, out_conv5_t1 = out_conv5

output_t0,output_t1,dist_flag = out_conv5_t0,out_conv5_t1,'l2'
interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
n, c, h, w = output_t0.data.shape
out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
similar_distance_map = distance.view(h,w).data.cpu().numpy()
similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))

#MaskLoss = ls.ConstractiveMaskLoss()
MaskLoss = ls.ConstractiveLoss()

contractive_loss_conv5 = MaskLoss(inputs1,inputs2,targets)

print (contractive_loss_conv5)

similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
plt.imshow(similar_dis_map_colorize)
plt.show()
