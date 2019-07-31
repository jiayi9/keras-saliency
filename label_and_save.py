#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:03:42 2019

@author: Lu Jiayi
"""


import os
import glob
from PIL import Image,ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
from keras.models import load_model
import matplotlib.cm as cm
import uuid


MODEL_PATH = ""
SOURCE_PATH = ""
OUTPUT_PATH = ""
FINAL_DENSE_LAYER_NAME = ""

MODEL_PATH = "/Users/tef-itm/Downloads/96_5cm.h5"
SOURCE_PATH = ""
OUTPUT_PATH = ""
FINAL_DENSE_LAYER_NAME = "dense_10"

print(MODEL_PATH, SOURCE_PATH, OUTPUT_PATH, FINAL_DENSE_LAYER_NAME)

LABEL = {
   "PP_valve_adpater":0,
   "Backflow_line":1,
   "Equalizing_element":2,
   "Housing":3,
   "Main_filter":4,
   "MP_valve_module":5,
   "PP_menbrane":6
}

SWAP_SOFTMAX_TO_LINEAR = False

###############################################################################

# load model
model = load_model(MODEL_PATH)

# Swap softmax with linear
if SWAP_SOFTMAX_TO_LINEAR:
    model.layers[11].activation = activations.linear
    model = utils.apply_modifications(model)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, FINAL_DENSE_LAYER_NAME)

# check layers in the model
NAMES = []
for index, layer in enumerate(model.layers):
    NAMES.append(layer.name)
    print(index, layer.name)
    
print('====================================================\n\n')



# label and save a single image
def label_and_save(model, source, output, class_index):
    img = np.array(Image.open(source))/255
    grads = visualize_saliency(model, 
                               layer_idx, 
                               filter_indices = CLASS_INDEX,
                               seed_input=img, 
                               backprop_modifier="guided")
    # 2
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(grads)
    rgb_img = np.delete(rgba_img, 3, 2)
    
    #3 
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    Overlay = overlay(jet_heatmap, img*255)
    R = np.concatenate((img, rgb_img, Overlay/255), axis = 1)
    
    im = Image.fromarray(np.uint8(R*255))
    
    uuid_str = uuid.uuid4().hex
    im.save(output+uuid_str+'.jpg')
#label_and_save(model, SOURCE_FOLDER+'0_0.jpg', OUTPUT_FOLDER, CLASS_INDEX)


# label and save a single image for all class
def label_and_save_contrast(model, source, output, CLASS_NUM):
    img = np.array(Image.open(source))/255
    GRADS = []
    MAX = np.array([])
    
    for index in range(CLASS_NUM):
        grads = visualize_saliency(model, layer_idx, filter_indices=index,
                               seed_input=img, backprop_modifier="guided")
        MAX = np.append(MAX, grads.max())
        GRADS.append(grads)
        
    MAX_PIXEL = MAX.max()
    
    L = []
    
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
        draw.text((0 + 5, i*96 + 5),  str(i), fill = (255, 255, 255))
    
    uuid_str = uuid.uuid4().hex
    im.save(output+uuid_str+'.jpg')

#source = SOURCE_FOLDER + '0_0.jpg'
#output = "/Users/tef-itm/Documents/label_and_save/OUTPUT/Backflow_line"
#label_and_save_contrast(model, source, output, CLASS_NUM = 7)


FILE_FORMAT = '*.jpg'
CLASS_INDEX = 7

SOURCE_FOLDER = "/Users/tef-itm/Documents/label_and_save/compressed_data_96/Backflow_line/"
OUTPUT_FOLDER = "/Users/tef-itm/Documents/label_and_save/Output/Backflow_line/"

files = glob.glob(SOURCE_FOLDER+FILE_FORMAT)

for index, file in enumerate(files):
    print(index, file)
    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_INDEX)



SOURCE_FOLDER = "/Users/tef-itm/Documents/label_and_save/compressed_data_96/Main_filter/"
OUTPUT_FOLDER = "/Users/tef-itm/Documents/label_and_save/Output/Main_filter/"
files = glob.glob(SOURCE_FOLDER+FILE_FORMAT)

for index, file in enumerate(files):
    print(index, file)
    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_INDEX)

    

SOURCE_FOLDER = "/Users/tef-itm/Documents/label_and_save/compressed_data_96/Equalizing_element/"
OUTPUT_FOLDER = "/Users/tef-itm/Documents/label_and_save/Output/Equalizing_element/"
files = glob.glob(SOURCE_FOLDER+FILE_FORMAT)

for index, file in enumerate(files):
    print(index, file)
    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_INDEX)


SOURCE_FOLDER = "/Users/tef-itm/Documents/label_and_save/compressed_data_96/Housing/"
OUTPUT_FOLDER = "/Users/tef-itm/Documents/label_and_save/Output/Housing/"
files = glob.glob(SOURCE_FOLDER+FILE_FORMAT)

for index, file in enumerate(files):
    print(index, file)
    label_and_save_contrast(model, file, OUTPUT_FOLDER, CLASS_INDEX)








