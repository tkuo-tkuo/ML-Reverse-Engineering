# create & store the dataset for our reverse model
import general_utils, database_utils, reverse_model
import numpy as np
database_utils = database_utils.DatabaseGenerationUtils()
utils = general_utils.GeneralUtils()

# create & store the dataset for our reverse model
# database_utils.generate_database_samples(200, 20, 'weight_database', 'attention_database')
weights_database, attention_database = database_utils.load_database_samples('weight_database', 'attention_database')
print(weights_database.shape, attention_database.shape)


# extract dataset
weights_database, attention_database = database_utils.load_database_samples('weight_database', 'attention_database')
number_of_train_datas = 160
train_datas = attention_database[:number_of_train_datas]
train_datas = utils.normalize(train_datas)
test_datas = attention_database[number_of_train_datas:]
test_datas = utils.normalize(test_datas)
train_labels = weights_database[:number_of_train_datas]
train_labels = utils.normalize(train_labels)
test_labels = weights_database[number_of_train_datas:]
test_labels = utils.normalize(test_labels)

bias = 0.2
shrink_rate = 0.00005
train_datas = train_datas 
test_datas = test_datas 
train_labels = train_labels * shrink_rate + bias
test_labels = test_labels * shrink_rate + bias
print(train_labels[0][0])

train_labels = train_labels[:, :10]
test_labels = test_labels[:, :10]
print(train_labels.shape, test_labels.shape)


# train our reverse model by dataset 
network = reverse_model.FCNetwork()
network.load_data(train_datas, train_labels, test_datas, test_labels)

model = network.create_simple_FC_model()
model = network.compile_model(model)
model = network.train_model(model, train_datas, train_labels, verbose=True)

results = model.predict(test_datas)
results = results - bias
test_labels = test_labels - bias
print(test_labels.shape, results.shape)
distance = np.absolute(test_labels - results) 
avg_distance = np.mean(distance)
mean_true_labels = np.mean(np.absolute(test_labels))

distribution_proporiton = avg_distance / mean_true_labels
print(distribution_proporiton)

print(results[0][5], test_labels[0][5])

# evaluate the performace by our self-defined metrics (design a function to produce practical metric)


# next object -> just totally flatten it by travesing the whole architecture ()

# Current problem: it's really hard for training
'''
1. First trail to cope with the problem: multiply both training set labels and test set labels to faciliate gradient descent 
(it did provide some improvement)
2. print out maybe every 10 steps instead of every step
3. draw a training curve 

4. maybe it's the problem of softmax -> it will enlarge the difference and let some of values approach to 0
5. we our prediction labels should not be a mixture of negative and positive values
6. maybe change the model architecture. For instance, CNN. 
Must:
a. extract the code in usage.py into different function
b. write code to faciliate the training process
'''