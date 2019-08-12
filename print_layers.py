
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import models
import numpy as np

import glob
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
#from keras import activations
from keras.models import load_model
import matplotlib.cm as cm
import uuid

def print_hidden_layers(image_path, model, images_per_row = 16, MAX_LAYER_INDEX = 10):
    img = image.load_img(image_path)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /= 255

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    classes = model.predict_classes(images)

    print(classes)

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    classifier = model
    for layer in classifier.layers[:MAX_LAYER_INDEX]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot    
    
#    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#        print(layer_name, layer_activation)

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        # n_rows
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix   
 #       n_cols = int(np.ceil(n_features/images_per_row))
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        # col is row, row is col
        for col in range(n_cols): # Tiles each filter into a big horizontal grid    
            for row in range(images_per_row):   
                if col * images_per_row + row < n_features:
                    channel_image = layer_activation[0,    
                                                     :, :,    
                                                     col * images_per_row + row]    
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable                    channel_image /= channel_image.std()    
                    channel_image *= 64    
                    channel_image += 128    
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')    
                    display_grid[col * size : (col + 1) * size, # Displays the grid    
                                 row * size : (row + 1) * size] = channel_image    
                else:
                    pass
        scale = 1. / size    
        plt.figure(figsize=(scale * display_grid.shape[1],    
                            scale * display_grid.shape[0]))    
        plt.title(layer_name)    
        plt.grid(False)    
        plt.imshow(display_grid, aspect='auto', cmap='viridis')




# load model
MODEL_PATH = "/Users/tef-itm/Downloads/96_5cm.h5"
model = load_model(MODEL_PATH)
SOURCE_FOLDER = "/Users/tef-itm/Documents/label_and_save/compressed_data_96/Backflow_line/"
OUTPUT_FOLDER = "/Users/tef-itm/Documents/label_and_save/Output/Backflow_line/"

FILE_FORMAT = '*.jpg'
files = glob.glob(SOURCE_FOLDER + FILE_FORMAT)

image_path = files[0]

print_hidden_layers(image_path, model,16,8)
