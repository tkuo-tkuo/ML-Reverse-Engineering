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

    def normalize(self, samples):
        mean = samples.mean().astype(np.float32)
        std = samples.std().astype(np.float32)
        normalized_samples = (samples - mean) / (std)
        return normalized_samples