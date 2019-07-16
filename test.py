# Load packages which are frequently used through experiments
import numpy as np
import matplotlib.pyplot as plt
import random, math

import Reverser

weightmodel_architecture = 'FC'
num_of_model_extracted_for_training = 100
num_of_model_extracted_for_testing = 30 
batch_size = 10
num_of_epochs = 20
num_of_print_interval = 10

interface = Reverser.ExperimentInterface(
    weightmodel_architecture,
    num_of_model_extracted_for_training, 
    num_of_model_extracted_for_testing, 
    batch_size, num_of_epochs, num_of_print_interval)

interface.train_weightmodel()
interface.test_weightmodel()
