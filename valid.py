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
    import dataset.TSUNAMI as dates
if datasets == 'GSV':
    import cfgs.GSVconfig as cfg
    import dataset.GSV as dates
if datasets == 'CMU':
    import cfgs.CMUconfig as cfg
    import dataset.CMU as dates
if datasets == 'CD2014':
    import cfgs.CD2014config as cfg
    import dataset.CD2014 as dates

resume = 1

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

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        #os.mkdir(dir)

def set_base_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' not in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_2x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' not in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def set_10x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_20x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def untransform(transform_img,mean_vector):

    transform_img = transform_img.transpose(1,2,0)
    transform_img += mean_vector
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance


testCase1_01 = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/t0/62.jpg'
testCase1_02 = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/t1/62.jpg'
testCase1_bg = '/home/m0729013/SceneChangeDet/dataset/cd2014/dataset/pic/bg/62.png'


img1 = Image.open(testCase1_01)
img2 = Image.open(testCase1_02)
imgbg = Image.open(testCase1_bg)

img1 = img1.resize((570,340),Image.ANTIALIAS)
img2 = img2.resize((570,340),Image.ANTIALIAS)
imgbg = imgbg.resize((570,340),Image.ANTIALIAS)

height,width,_ = np.array(img1,dtype= np.uint8).shape
img1 = np.array(img1,dtype= np.uint8)
img2 = np.array(img2,dtype= np.uint8)
#label = np.zeros((height,width,3),dtype=np.uint8) ###################
label = np.array(imgbg,dtype=np.uint8)

img1, img2, label = data_transform(img1, img2, label)
inputs1,input2, targets = img1, img2, label
model = models.SiameseNet(norm_flag='l2')

pretrain_deeplab_path = "/home/m0729013/SceneChangeDet/code/pretrain/negative2.pth"
deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
model.init_parameters_from_deeplab(deeplab_pretrain_model)

model = model.cuda()
model.eval()

cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
metric_for_conditions = util.init_metric_for_class_for_cmu(1)

inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0), volatile=True),Variable(input2.unsqueeze(0),volatile=True) ,Variable(targets)
out_conv5,out_fc,out_embedding = model(inputs1,inputs2)

out_conv5_t0, out_conv5_t1 = out_fc

output_t0,output_t1,dist_flag = out_conv5_t0,out_conv5_t1,'l2'

interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
n, c, h, w = output_t0.data.shape
out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)

print(distance)

similar_distance_map = distance.view(h,w).data.cpu().numpy()


#K=100
MAX_3 = np.argpartition(similar_distance_map.ravel(),-3212)[-3212:]


print(np.mean(similar_distance_map))
print(np.max(similar_distance_map))
#print(MAX_3)
#print(np.mean(MAX_3))

similar_distance_map = similar_distance_map/np.max(similar_distance_map)
#print(similar_distance_map)

MaskLoss = ls.ConstractiveMaskLoss()
MaskLoss = ls.ConstractiveLoss()

contractive_loss_conv5 = MaskLoss(inputs1,inputs2,targets)

print (contractive_loss_conv5)

#contractive_loss_fc = MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
#contractive_loss_embedding = MaskLoss(out_embedding_t0,out_embedding_t1,label_rz_embedding)

'''
for i in range(h):
    for j in range(w):
        if similar_distance_map[i][j] < 0.75:
            similar_distance_map[i][j] = 0
        else:
            similar_distance_map[i][j] = 1
'''
  
#print(similar_distance_map)

similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))

#print(similar_distance_map_rz.data.cpu().numpy()[0][0])

similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_RAINBOW)
plt.imshow(similar_dis_map_colorize)
plt.show()

