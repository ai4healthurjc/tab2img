import numpy as np
import cv2


class Interpolation:
    def __init__(self, image, scale=5, type_interpolation='nearest'):

        self.image = image
        self.image_width = self.image.shape[0]
        self.image_height = self.image.shape[1]
        # self.resized_image_width = self.image_width * scale
        # self.resized_image_height = self.image_height * scale
        self.type_interpolation = type_interpolation
        self.scale = scale

    def compute_interpolation(self):

        kernel_size = self.scale

        if self.type_interpolation == 'average':
            return cv2.blur(self.image, (kernel_size, kernel_size))
        elif self.type_interpolation == 'median':
            return cv2.medianBlur(self.image, kernel_size)
        elif self.type_interpolation == 'gaussian':
            return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif self.type_interpolation == 'bilateral':
            return cv2.bilateralFilter(self.image, kernel_size, 75, 75)
        else:
            print('Interpolation method not found!')

