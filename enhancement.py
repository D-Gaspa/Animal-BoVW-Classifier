import cv2 as cv
import skimage
from brisque import BRISQUE
import random
import numpy as np
import matplotlib.pyplot as plt

print("Image Enchancement:\n")

dspath = ""
animal_arr = np.empty(50)
animal_imgs = np.empty(50)
scoreBrisqueVal =  np.empty(50)
BrisqueVal = BRISQUE(url = False)
for i in range(50):
    ran = random.randint(1, 5)
    animal_arr[i] = ran
    dspath = "./data/raw_dataset"
    
    if ran == 1:
        dspath = dspath + "/Crocodile"
        ran = random.randint(1, 100)
        animal_imgs[i] = ran
        dspath = dspath + f"/c{ran}.jpg"
    elif ran == 2:
        dspath = dspath + "/Fox"
        ran = random.randint(1, 100)
        animal_imgs[i] = ran
        dspath = dspath + f"/f{ran}.jpg"
    elif ran == 3:
        dspath = dspath + "/Giraffe"
        ran = random.randint(1, 100)
        animal_imgs[i] = ran
        dspath = dspath + f"/g{ran}.jpg"
    elif ran == 4:
        dspath = dspath + "/Panda"
        ran = random.randint(1, 100)
        animal_imgs[i] = ran
        dspath = dspath + f"/p{ran}.jpg"
    elif ran == 5:
        dspath = dspath + "/Raccoon"
        ran = random.randint(1, 100)
        animal_imgs[i] = ran
        dspath = dspath + f"/r{ran}.jpg"
    img = cv.imread(dspath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    scoreBrisqueVal[i] = BrisqueVal.score(img)
avg = np.sum(scoreBrisqueVal) / 50
print(f"The BRISQUE average of 50 random images is: {avg}")


    

    
    
