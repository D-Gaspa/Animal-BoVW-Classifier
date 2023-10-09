import cv2
import numpy as np

class Filters:
    def __init__(self):
        pass

    @staticmethod
    def histogram_equalization(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def clahe(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = cv2.split(image)
        channels = [clahe.apply(channel) for channel in channels]
        return cv2.merge(channels)

    @staticmethod
    def edge_enhancement(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def noise_reduction(image):
        enhance = cv2.bilateralFilter(image, 9, 75, 75)
        return enhance
        #return cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

    @staticmethod
    def sharpen(image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def canny(image):
       gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
       edges = cv2.Canny(gray, 120, 250, L2gradient= True)
       edges = cv2.Laplacian(image, cv2.CV_64F)
       return edges
    
    @staticmethod
    def unsharp(image):
        enhance = cv2.GaussianBlur(image, (0,0), 2.0)
        enhance = cv2.addWeighted(image, 2.0, enhance, -1.0, 0)
        return enhance

    @staticmethod
    def threshold(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhance = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
        return enhance
    
    @staticmethod
    def gabor(image):
        filters = []
        num_filters = 16
        ksize = 30  # The local area to evaluate
        sigma = 3.0  # Larger Values produce more edges
        lambd = 10.0
        gamma = 0.5
        psi = 0  # Offset value - lower generates cleaner results
        for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
            kern /= 1.0 * kern.sum()  # Brightness normalization
            filters.append(kern)
        newimage = np.zeros_like(image)
        depth = -1 # remain depth same as original image
     
        for kern in filters:  # Loop through the kernels in our GaborFilter
            image_filter = cv2.filter2D(image, depth, kern)  #Apply filter to image
            
            # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
            np.maximum(newimage, image_filter, newimage)
        #enhance = cv2.getGaborKernel((35, 35), sigma= 3, theta= 0, lambd=10, psi= 0, gamma=0.5)
        #res = cv2.filter2D(image, ddepth= cv2.CV_32F, kernel=enhance)
        return newimage
        
       
