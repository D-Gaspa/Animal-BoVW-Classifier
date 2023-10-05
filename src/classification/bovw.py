import numpy as np
import cv2 as cv
import os
import pprint
from pathlib import Path
from sklearn.model_selection import train_test_split


data_path = f'{Path.cwd()}\\data\\enhanced_images'
categories = []
data = []
labels = []
descriptors = []


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
            
data = np.asarray(data)  
labels = np.asarray(labels)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size= 0.3, train_size= 0.7, shuffle= True, random_state = 35, stratify=labels)

orb = cv.ORB.create()

kp, des = orb.detectAndCompute()

for d in des:
    descriptors.append(d)
print(labels)


