import numpy as np
import matplotlib.pyplot as plt


class GeneralUtils():

    def __init__(self):
        pass

    def print_image_with_1D_array(self, image):
        # Assume image is with 784 pixels
        image = np.array(image, dtype='float')
        pixels = image.reshape((28, 28))
        plt.imshow(pixels)
        plt.show()