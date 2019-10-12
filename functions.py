#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For Windows only because of font issue. Modify if you need it in any Linux systems.
# There are three functions in this script
# 1. visualize_saliency_gray_linear:   For one image, create three images [raw, saliency, overlay] with linear activation for last layer
# 2. visualize_saliency_gray:          For one image, create three*classes iamges [raw, saliency, overlay]*classes with softmax activation for last layer for CONTRAST!create three*classes iamges [raw, saliency, overlay]*classes with softmax activation for last layer for CONTRAST!
# 3. label_and_save_contrast:          For RGB or gray images with softmax activation for last layer 

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
import cv2

# gray linear

def visualize_saliency_gray_linear(model, file_folder, output_folder):
    layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
    CLASS_NUM = model.layers[-1].output_shape[1]
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    
    width = model.layers[0].input.get_shape()[1].value
    height= model.layers[0].input.get_shape()[2].value
    assert(width == height)
    
    img = np.array(Image.open(file_path))/255
    img = np.expand_dims(img,2)
    img = np.expand_dims(img,0)         
    img_rgb_bg = np.array(Image.open(file_path).convert("RGB"))/255     
   
    print(model.predict_classes(img))
    prob = model.predict_proba(img).round(3)
    print(prob)

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
    for i, grads in enumerate([GRADS[0]]):
            # 2
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
    # vertical lines
    for i in range(3):
        TMP = cv2.line(TMP, (i*width,0), (i*width ,width*3),  color = (1, 1, 1), thickness = 1)

    im = Image.fromarray(np.uint8(TMP*255))
#    draw = ImageDraw.Draw(im)
#    for i in range(CLASS_NUM):
#        draw.text((0 + 5, i*width + 5),  str(i), fill = text_color, font=Font1)

    uuid_str = uuid.uuid4().hex
    im.save(output_folder+uuid_str+'.jpg')




# gray 
def visualize_saliency_gray(model, file_folder, output_folder):
    Font1 = ImageFont.truetype("C:\Windows\Fonts\simsunb.ttf",36)
    
    text_color = (255,255,255)
    
    layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
    CLASS_NUM = model.layers[-1].output_shape[1]
    
    width = model.layers[0].input.get_shape()[1].value
    height= model.layers[0].input.get_shape()[2].value
    assert(width == height)
    
    img = np.array(Image.open(file_path))/255
    img = np.expand_dims(img,2)
    img = np.expand_dims(img,0)         
    img_rgb_bg = np.array(Image.open(file_path).convert("RGB"))/255     
   
    print(model.predict_classes(img))
    prob = model.predict_proba(img).round(3)
    print(prob)

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
    # vertical lines
    for i in range(3):
        TMP = cv2.line(TMP, (i*width,0), (i*width ,width*3),  color = (1, 1, 1), thickness = 1)
    # horizontal lines
    for i in range(CLASS_NUM):
        TMP = cv2.line(TMP, (0, i*width), (width*3, i*width),  color = (1, 1, 1), thickness = 1)

    im = Image.fromarray(np.uint8(TMP*255))
    draw = ImageDraw.Draw(im)
    for i in range(CLASS_NUM):
        draw.text((0 + 5, i*width + 5),  str(i), fill = text_color, font=Font1)
        draw.text((0 + 5 + width, i*width + 5), str(prob[0][i]), fill = text_color, font=Font1)

    uuid_str = uuid.uuid4().hex
    im.save(output_folder+uuid_str+'.jpg')




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
        prob = model.predict_proba(img).round(3)
        print(prob)
        
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


        
    elif COLOR_MODE == "L":
        img = np.array(Image.open(file_path))/255
        img = np.expand_dims(img,2)
        img = np.expand_dims(img,0)         
        img_rgb_bg = np.array(Image.open(file_path).convert("RGB"))/255     
        
        print(model.predict_classes(img))
        prob = model.predict_proba(img).round(3)
        print(prob)

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

    for i in range(3):
        TMP = cv2.line(TMP, (i*width,0), (i*width ,width*3),  color = (1, 1, 1), thickness = 1)
    for i in range(CLASS_NUM):
        TMP = cv2.line(TMP, (0, i*width), (width*3, i*width),  color = (1, 1, 1), thickness = 1)

    im = Image.fromarray(np.uint8(TMP*255))
    draw = ImageDraw.Draw(im)
    for i in range(CLASS_NUM):
        draw.text((0 + 5, i*width + 5),  str(i), fill = text_color, font=Font1)
        draw.text((0 + 5 + width, i*width + 5), str(prob[0][i]), fill = text_color, font=Font1)
    
    uuid_str = uuid.uuid4().hex
    im.save(output_folder+uuid_str+'.jpg')

    
    
 

##---------------------------    test code    ----------------------------------
#
## load model
##MODEL_PATH = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/simulation/model.h5"
#
#MODEL_PATH = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/simulation/model_5Con.h5"
#
#
#model = load_model(MODEL_PATH)
##
## swap softmax
##
#layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
#CLASS_NUM = model.layers[-1].output_shape[1]
#
#
##model_2 = model
##SWAP_SOFTMAX_TO_LINEAR = False
##if SWAP_SOFTMAX_TO_LINEAR:
##    model_2.layers[layer_idx].activation = activations.linear
##    model_2 = utils.apply_modifications(model_2)
##
### check layers in the model
#NAMES = []
#for index, layer in enumerate(model.layers):
#    NAMES.append(layer.name)
#    print(index, layer.name)
#print('\n\n')
#
#
##  line
#SOURCE_FOLDER = "C:/LV_CHAO_IMAGE/simulation_data/line_256/"
#OUTPUT_FOLDER = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/simulation/output/line/"
#
#FILE_FORMAT = '*.jpg'
#files = glob.glob(SOURCE_FOLDER + FILE_FORMAT)
#
#for index, file in enumerate(files[0:50]):
#    print(index, file)
#    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_NUM, layer_idx, width = 256, COLOR_MODE = "L")
#
#
#
#
##  circle 
#SOURCE_FOLDER = "C:/LV_CHAO_IMAGE/simulation_data/circle_256/"
#OUTPUT_FOLDER = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/simulation/output/circle/"
#
#FILE_FORMAT = '*.jpg'
#files = glob.glob(SOURCE_FOLDER + FILE_FORMAT)
#
#for index, file in enumerate(files[0:50]):
#    print(index, file)
#    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_NUM, layer_idx, width = 256, COLOR_MODE = "L", text_color = (255,255,255))
#
#
#
#
## none
#SOURCE_FOLDER = "C:/LV_CHAO_IMAGE/simulation_data/pass_256/"
#OUTPUT_FOLDER = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/simulation/output/none/"
#
#FILE_FORMAT = '*.jpg'
#files = glob.glob(SOURCE_FOLDER + FILE_FORMAT)
#
#for index, file in enumerate(files[0:50]):
#    print(index, file)
#    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_NUM, layer_idx, width = 256, COLOR_MODE = "L")

