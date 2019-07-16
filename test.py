# Load packages which are frequently used through experiments
import numpy as np
import matplotlib.pyplot as plt
import random, math

import Reverser

num_of_model_extracted_for_training = 100
num_of_model_extracted_for_testing = 30 
batch_size = 10
num_of_epochs = 10
num_of_print_interval = 10

interface = Reverser.ExperimentInterface(
    num_of_model_extracted_for_training, 
    num_of_model_extracted_for_testing, 
    batch_size, num_of_epochs, num_of_print_interval)

# interface.train_weightmodel()
