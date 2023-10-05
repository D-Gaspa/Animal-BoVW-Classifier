import cv2
import numpy as np


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def histogram_equalization(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

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
        return cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

    @staticmethod
    def sharpen(image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
