import numpy as np

class FeatureExtrationUtils():

    def __init__(self):
        pass

    def extract_weights(self, model):
        weights = []
        layers = [l for l in model.layers]
        for layer in layers:
            weights.append(layer.get_weights())
        return np.array(weights)

    def extract_attention(self, model, image, label):
        '''
        This function assumes that the image is resized to be consistent with the given model
        '''
        length = len(image)
        sensitivity_image = []
        for i in range(length):
            pixel_sensitivity = self.get_sensitiy_of_certain_pixel(model, image, label, i)
            sensitivity_image.append(pixel_sensitivity)
        
        return np.array(sensitivity_image)

    def evaluate_simliarity(self, actual_model, predict_model):
        pass
        
    def get_sensitiy_of_certain_pixel(self, model, image, label, index):
        pos_sensitivity = self.get_sensitiy_of_certain_pixel_on_single_direction(model, image, label, index, True)
        neg_sensitivity = self.get_sensitiy_of_certain_pixel_on_single_direction(model, image, label, index, False)
        avg_sensitivity = (pos_sensitivity + neg_sensitivity) / 2
        return avg_sensitivity 

    def get_sensitiy_of_certain_pixel_on_single_direction(self, model, image, label, index, isPositive):
        diff_unit = 0.001
        if not isPositive:
            diff_unit *= -1
            
        scale = 1
        total_traversed_unit = 0
        upper_bound_on_total_traversed_unit = 100000000
        image = np.expand_dims(image, axis=0)
        image_copy = np.copy(image)

        while total_traversed_unit < upper_bound_on_total_traversed_unit:
            image_copy[0][index] = image_copy[0][index] + (scale * diff_unit)

            prediction_values = model.predict(image_copy)[0]
            prediction, expected_prediction = np.argmax(prediction_values), np.argmax(label)
            isSame = (prediction == expected_prediction)
            total_traversed_unit += scale
            scale *= 2

            if not isSame:
                break

        abs_distance = abs(total_traversed_unit * diff_unit)
        sensitivity = 1/abs_distance
        return sensitivity

        