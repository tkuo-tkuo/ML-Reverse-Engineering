# Load packages which are frequently used through experiments
import numpy as np
import matplotlib.pyplot as plt
import random, math

import Reverser

weightmodel_architecture = 'FC'
num_of_model_extracted_for_training = 10 
num_of_model_extracted_for_testing = 1
batch_size = 1
num_of_epochs = 1
num_of_print_interval = 10 
lr = 1e-6

interface = Reverser.ExperimentInterface(
    weightmodel_architecture,
    num_of_model_extracted_for_training, 
    num_of_model_extracted_for_testing, 
    batch_size, num_of_epochs, lr, num_of_print_interval)

# interface.train_weightmodel()
# interface.test_weightmodel()
interface.verify_weightmodel_reverse_effectiveness()
interface.generate_substitute_model()
