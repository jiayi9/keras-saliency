#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from PIL import Image, ImageDraw,ImageFont
import numpy as np
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
from keras.models import load_model
import matplotlib.cm as cm
import uuid

# label and save a single image for all class
def label_and_save_contrast(model, file_path, output_folder, CLASS_NUM, layer_idx, width = 96, COLOR_MODE = "RGB", text_color = (255,255,255)):
    
    Font1 = ImageFont.truetype("C:\Windows\Fonts\simsunb.ttf",36)

    if COLOR_MODE == "RGB":
    
        img = np.array(Image.open(file_path))/255
        GRADS = []
        MAX = np.array([])    
        for index in range(CLASS_NUM):
            grads = visualize_saliency(model, layer_idx, filter_indices=index,
                                   seed_input=img, backprop_modifier="guided")
            MAX = np.append(MAX, grads.max())
            GRADS.append(grads)        
        MAX_PIXEL = MAX.max()
        L = []
        
        print(model.predict_classes(img))

        if MAX_PIXEL == 0:
            MAX_PIXEL = 0.00001

        for i, grads in enumerate(GRADS):
            # 2
            grads = grads/MAX_PIXEL        
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(grads)
            rgb_img = np.delete(rgba_img, 3, 2)            
            #3 
            jet_heatmap = np.uint8(cm.jet(grads)[...,:3] * 255)
            Overlay = overlay(jet_heatmap, img*255)
            L.append(np.concatenate((img, rgb_img, Overlay/255), axis = 1))
        TMP = []
        for i, IM in enumerate(L):
            if i == 0:
                TMP = IM
            else:
                TMP = np.concatenate((TMP, IM), axis = 0)
        im = Image.fromarray(np.uint8(TMP*255))
        draw = ImageDraw.Draw(im)
        for i in range(CLASS_NUM):
            draw.text((0 + 5, i*width + 5),  str(i), fill = text_color)
        uuid_str = uuid.uuid4().hex
        im.save(output_folder+uuid_str+'.jpg')
        
    elif COLOR_MODE == "L":
        img = np.array(Image.open(file_path))/255
        img = np.expand_dims(img,2)
        img = np.expand_dims(img,0)         
        img_rgb_bg = np.array(Image.open(file_path).convert("RGB"))/255     
        
        print(model.predict_classes(img))
        
        GRADS = []
        MAX = np.array([])        
        for index in range(CLASS_NUM):
            grads = visualize_saliency(model, layer_idx, filter_indices=index,
                                   seed_input=img, backprop_modifier="guided")
            MAX = np.append(MAX, grads.max())
            GRADS.append(grads)
        MAX_PIXEL = MAX.max()
        
        if MAX_PIXEL == 0:
            MAX_PIXEL = 0.00001
        
        L = []
        for i, grads in enumerate(GRADS):
            # 2
            #grads[0,0] = MAX_PIXEL
            grads = grads/MAX_PIXEL        
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(grads)
            rgb_img = np.delete(rgba_img, 3, 2)
            #3 
            jet_heatmap = np.uint8(cm.jet(grads)[...,:3] * 255)
            Overlay = overlay(jet_heatmap, img_rgb_bg*255)
            L.append(np.concatenate((img_rgb_bg, rgb_img, Overlay/255), axis = 1))
        TMP = []
        for i, IM in enumerate(L):
            if i == 0:
                TMP = IM
            else:
                TMP = np.concatenate((TMP, IM), axis = 0)
        im = Image.fromarray(np.uint8(TMP*255))
        draw = ImageDraw.Draw(im)
        
        
        for i in range(CLASS_NUM):
            # draw.text((0 + 5, i*width + 5),  str(i), fill = (255, 0, 255))

            draw.text((0 + 5, i*width + 5),  str(i), fill = (255, 0, 255), font=Font1)
        uuid_str = uuid.uuid4().hex
        im.save(output_folder+uuid_str+'.jpg')


#---------------------------    test code    ----------------------------------

# load model
#MODEL_PATH = "/Users/tef-itm/Downloads/96_5cm.h5"
#MODEL_PATH = "//bosch.com/dfsrb/DfsCN/LOC/Wx/Project/RBCD_Data_mining/Data_Analytics_Community/exchange/lv/models/96_5cm.h5"
#MODEL_PATH = "//bosch.com/dfsrb/DfsCN/LOC/Wx/Project/RBCD_Data_mining/Data_Analytics_Community/exchange/lv_moe9/saliency_checks/model_1/ValvePiece_ResNet56v2_model.022.h5"

MODEL_PATH = "C:/LV_CHAO_IMAGE/validate_lv/model.h5"

model = load_model(MODEL_PATH)
#
# swap softmax
#
layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
CLASS_NUM = model.layers[-1].output_shape[1]


#model_2 = model
#SWAP_SOFTMAX_TO_LINEAR = False
#if SWAP_SOFTMAX_TO_LINEAR:
#    model_2.layers[layer_idx].activation = activations.linear
#    model_2 = utils.apply_modifications(model_2)
#
## check layers in the model
NAMES = []
for index, layer in enumerate(model.layers):
    NAMES.append(layer.name)
    print(index, layer.name)
print('\n\n')
#

#SOURCE_FOLDER = "/Users/tef-itm/Documents/suphina/sample/"
#OUTPUT_FOLDER = "/Users/tef-itm/Documents/suphina/output/"
#SOURCE_FOLDER = "N:/RBCD_Data_mining/Data_Analytics_Community/exchange/lv/compressed_96_96_flip_2/Backflow_line/"
#OUTPUT_FOLDER = "N:/RBCD_Data_mining/Data_Analytics_Community/exchange/lv_moe9/saliency_checks/model_1/output/Good_2/"
SOURCE_FOLDER = "N:/RBCD_Data_mining/Data_Analytics_Community/exchange/lv_moe9/saliency_checks/model_1/sample_2/"
OUTPUT_FOLDER = "N:/RBCD_Data_mining/Data_Analytics_Community/exchange/lv_moe9/saliency_checks/model_1/output/Scratch_2/"

SOURCE_FOLDER = "C:/LV_CHAO_IMAGE/validate_lv/source/"
OUTPUT_FOLDER = "C:/LV_CHAO_IMAGE/validate_lv/output/"

FILE_FORMAT = '*.jpg'
files = glob.glob(SOURCE_FOLDER + FILE_FORMAT)

for index, file in enumerate(files):
    print(index, file)
    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_NUM, layer_idx, width = 256, COLOR_MODE = "L")


#---------------------------   more test code    ----------------------------------

#
#file_path = files[0]
#img = np.array(Image.open(file_path))/255
#
#plt.imshow(img)
#plt.imshow(img, cmap = "binary")
#plt.imshow(img, cmap = "gray")
#
#
#img = np.array(Image.open(file_path))/255
#img = np.expand_dims(img,2)
#img = np.expand_dims(img,0) 
#
#import time
#start = time.time()     
#print(model.predict_classes(img))
#end = time.time()
#print("Execution time: ",end - start)
