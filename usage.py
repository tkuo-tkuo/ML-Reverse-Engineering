# import all the basic settings or essentail files
import target_model, feature_extraction_utils, general_utils
import numpy as np
import csv
import pandas

network = target_model.FCNetwork()
feature_extraction_utils = feature_extraction_utils.FeatureExtrationUtils()
general_utils = general_utils.GeneralUtils()

# create a simple model and store our target model 
'''
model = network.train_and_save_simply_FC_model('model', verbose=True)
'''


with open('database.csv', mode='w') as csv_file:
    fieldnames = ['attention']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    
    (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
    number_of_samples_generated = 20

    for index in range(number_of_samples_generated):
        # load our target blackbox model 
        model = network.train_and_save_simply_FC_model('model', verbose=True)
        '''
        network.evaluate_model(model, test_datas, test_labels)
        '''

        # write a function called extract_weights weights given model in feature_extraction_utils.py
        '''
        weights = feature_extraction_utils.extract_weights(model)
        print(weights.shape) # it should display (3, 2)
        '''
        model = network.load_model('model')
        model.save_weights('model'+str(index+1)+'.h5')
        weights = feature_extraction_utils.extract_weights(model)
        

        # write a function called extract_attentions given image and model in feature_extraction_utils.py
        sensitivity_image = np.zeros_like(train_datas[0])
        number_of_samples = 10
        for i in range(number_of_samples):
            print(i)
            image_sample_data, image_sample_label = train_datas[i], train_labels[i]
            sensitivity_image_i = feature_extraction_utils.extract_attention(model, image_sample_data, image_sample_label)
            sensitivity_image = sensitivity_image + sensitivity_image_i

        sensitivity_image /= number_of_samples

        writer.writerow({'attention': sensitivity_image})








# store the dataset for our reverse model



# train our reverse model by dataset 



# evaluate the performace by our self-defined metrics (design a function to produce practical metric)


