# Load packages which are frequently used through experiments
import numpy as np
import matplotlib.pyplot as plt
import random, math

num_of_model_extracted = 30 
batch_size = 6
num_of_epochs = 50
num_of_print_interval = 5

import Reverser
interface = Reverser.ExperimentInterface(num_of_model_extracted, batch_size, num_of_epochs, num_of_print_interval)

interface.train_weightmodel()
