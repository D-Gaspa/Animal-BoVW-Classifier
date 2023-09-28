import cv2 as cv
import skimage
from skimage.transform import resize
from brisque import BRISQUE
import random
import numpy as np
import matplotlib.pyplot as plt
import glob

print("Image Enchancement:\n")

dspath = ""
animal_arr = np.empty(50)
animal_imgs = np.empty(50)
scoreBrisqueVal =  np.empty(50)
BrisqueVal = BRISQUE(url = False)

def BrisqueOriginal():
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
    return avg

OrBrisqueResAVG = BrisqueOriginal()
print(f"The BRISQUE average of 50 random images is: {OrBrisqueResAVG}")

dspath = "./data/raw_dataset"
training_data = []
t = 1
for imgFile in glob.iglob(f"{dspath}/*/*"):
    img1 = cv.imread(imgFile)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img = resize(img1, (1500,1500))
    score1 = BrisqueVal.score(img1)
    score2 = BrisqueVal.score(img)
    print(f"score1: {score1} \nscore2: {score2}")
    cv.imwrite()
    
    #training_data.append(img)
    
    if t ==1:
        plt.title("Before scale")
        plt.imshow(img1)
        plt.show()
        plt.title("After scale")
        plt.imshow(img)
        plt.show()
        break;
    
#training_data = np.array(training_data)