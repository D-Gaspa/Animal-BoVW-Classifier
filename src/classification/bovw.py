import numpy as np
import cv2 as cv
import os
import pprint
from pathlib import Path


data_path = f'{Path.cwd()}\\data\\enhanced_images'
categories = []
data = []
labels = []


def prep_data():
    global labels
    for categories_index, class_name in enumerate(os.listdir(data_path)):
        categories.append(class_name)
        image_path = os.path.join(data_path, class_name)
        for image_file in os.listdir(image_path):
            image = cv.imread(os.path.join(image_path, image_file))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            data.append(image)
            labels.append(categories_index)
            
    labels = np.asarray(labels)
        

prep_data()
print(labels)


