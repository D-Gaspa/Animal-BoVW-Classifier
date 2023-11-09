import cv2
import numpy as np
from rembg import remove


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

    @staticmethod
    def edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def color_filtering(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # These values can be adjusted depending on the specific color ranges of interest
        lower_bound = np.array([0, 40, 40])
        upper_bound = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        filtered = cv2.bitwise_and(image, image, mask=mask)
        return filtered

    @staticmethod
    def adaptive_thresholding(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def gabor(image):
        filters = []
        num_filters = 16
        k_size = 30  # The local area to evaluate
        sigma = 3.0  # Larger Values produce more edges
        lamb = 10.0
        gamma = 0.5
        psi = 0  # Offset value - lower generates cleaner results
        for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
            kern = cv2.getGaborKernel((k_size, k_size), sigma, theta, lamb, gamma, psi, ktype=cv2.CV_64F)
            kern /= 1.0 * kern.sum()  # Brightness normalization
            filters.append(kern)
        new_image = np.zeros_like(image)
        depth = -1  # remain depth same as original image

        for kern in filters:  # Loop through the kernels in our GaborFilter
            image_filter = cv2.filter2D(image, depth, kern)  # Apply filter to image

            # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
            np.maximum(new_image, image_filter, new_image)
        return new_image

    @staticmethod
    def background_removal(image):
        # Convert the image to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # Remove the background
        result = remove(image)

        # Convert the image back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)

        return result

    @staticmethod
    def canny_edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 120, 250, L2gradient=True)
        return edges

    @staticmethod
    def unsharp_masking(image):
        enhance = cv2.GaussianBlur(image, (0, 0), 2.0)
        enhance = cv2.addWeighted(image, 2.0, enhance, -1.0, 0)
        return enhance
