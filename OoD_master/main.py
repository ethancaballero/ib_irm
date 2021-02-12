import tensorflow as tf
import numpy as np
import argparse
#import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd
tf.compat.v1.enable_eager_execution()
import cProfile
from sklearn.model_selection import train_test_split
import copy as cp
from sklearn.model_selection import KFold
from datetime import date
import time

import wandb
import plotly

from ERM_IRM.data_construct import * ## contains functions for constructing data 
from ERM_IRM.IRM_methods import *    ## contains IRM and ERM methods 


n_trial =10
n_tr_list = [1000, 5000, 10000, 30000, 60000] # list of training sample sizes

k=0
K = len(n_tr_list)
ERM_model_acc = np.zeros((K,n_trial))
ERM_model_acc_nb = np.zeros((K,n_trial))
IRM_model_acc = np.zeros((K,n_trial))
IRM_model_acc_v = np.zeros((K,n_trial))
IRM_model_ind_v = np.zeros((K,n_trial))

ERM_model_acc1 = np.zeros((K,n_trial))
ERM_model_acc1_nb = np.zeros((K,n_trial))
IRM_model_acc1 = np.zeros((K,n_trial))
IRM_model_acc1_v = np.zeros((K,n_trial))
IRM_model_ind_v = np.zeros((K,n_trial))

ERM_model_acc_av = np.zeros(K)
ERM_model_acc_av_nb = np.zeros(K)
IRM_model_acc_av = np.zeros(K)
IRM_model_acc_av_v = np.zeros(K)


ERM_model_acc_av1 = np.zeros(K)
ERM_model_acc_av1_nb = np.zeros(K)
IRM_model_acc_av1 = np.zeros(K)
IRM_model_acc_av1_v = np.zeros(K)

list_params = []
for n_tr in n_tr_list:
    print ("tr" + str(n_tr))
    t_start = time.time()
    for trial in range(n_trial):
        print ("trial " + str(trial))
        n_e=2
        p_color_list = [0.2, 0.1]
        p_label_list = [0.25]*n_e
        D = assemble_data_mnist_sb(n_tr) # initialize mnist digits data object

        D.create_training_data(n_e, p_color_list, p_label_list) # creates the training environments

        p_label_test = 0.25 # probability of switching pre-label in test environment
        p_color_test = 0.9  # probability of switching the final label to obtain the color index in test environment

        D.create_testing_data(p_color_test, p_label_test, n_e)  # sets up the testing environment
        (num_examples_environment,length, width, height) = D.data_tuple_list[0][0].shape # attributes of the data
        num_classes = len(np.unique(D.data_tuple_list[0][1])) # number of classes in the data

        model_erm =  keras.Sequential([
                keras.layers.Flatten(input_shape=(length,width,height)),
                keras.layers.Dense(390, activation = 'relu',kernel_regularizer=keras.regularizers.l2(0.0011)),
                keras.layers.Dense(390, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0011)),
                keras.layers.Dense(2, activation='softmax')
        ])
        
        num_epochs = 100
        batch_size = 512
        learning_rate = 4.9e-4
        erm_model1 = standard_erm_model(model_erm, num_epochs, batch_size, learning_rate)
        erm_model1.fit(D.data_tuple_list)
        erm_model1.evaluate(D.data_tuple_test)
        print ("Training accuracy:" + str(erm_model1.train_acc))
        print ("Testing accuracy:" + str(erm_model1.test_acc))
        
        ERM_model_acc[k][trial] = erm_model1.test_acc
        ERM_model_acc1[k][trial] = erm_model1.train_acc


        gamma_list = [10000, 33000, 66000,100000.0]
        index=0
        best_err = 1e6
        train_list =[]
        val_list = []
        test_list = []
        for gamma_new in gamma_list:

            model_irm = keras.Sequential([
                                keras.layers.Flatten(input_shape=(length,width,height)),
                                keras.layers.Dense(390, activation = 'relu',kernel_regularizer=keras.regularizers.l2(0.0011)),
                                keras.layers.Dense(390, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0011)),
                                keras.layers.Dense(num_classes)
                        ])
            batch_size       = 512
            steps_max        = 1000
            steps_threshold  = 190  ## threshold after which gamma_new is used
            learning_rate    = 4.9e-4


            irm_model1 = irm_model(model_irm, learning_rate, batch_size, steps_max, steps_threshold, gamma_new)
            irm_model1.fit(D.data_tuple_list)
            irm_model1.evaluate(D.data_tuple_test)
            error_val = 1-irm_model1.val_acc
            train_list.append(irm_model1.train_acc)
            val_list.append(irm_model1.val_acc)
            test_list.append(irm_model1.test_acc)
            if(error_val<best_err):
                index_best =index
                best_err = error_val
            index= index+1

        print ("Training accuracy:" + str(train_list[index_best]))
        print ("Validation accuracy:" + str(val_list[index_best]))
        print ("Testing accuracy:" + str(test_list[index_best]))

        IRM_model_acc_v[k][trial]  = test_list[index_best]
        IRM_model_acc1_v[k][trial] = train_list[index_best]
        IRM_model_ind_v[k][trial]  = index_best


    IRM_model_acc_av_v[k] = np.mean(IRM_model_acc_v[k])
    list_params.append([n_tr,"IRMv_test", np.mean(IRM_model_acc_v[k]),np.std(IRM_model_acc_v[k])])

    ERM_model_acc_av[k] = np.mean(ERM_model_acc[k])
    list_params.append([n_tr,"ERM_test", np.mean(ERM_model_acc[k]),np.std(ERM_model_acc[k])])


    IRM_model_acc_av1_v[k] = np.mean(IRM_model_acc1_v[k])
    list_params.append([n_tr,"IRMv_train", np.mean(IRM_model_acc1_v[k]),np.std(IRM_model_acc1_v[k])])
    
    ERM_model_acc_av1[k] = np.mean(ERM_model_acc1[k])
    list_params.append([n_tr, "ERM_train", np.mean(ERM_model_acc1[k]),np.std(ERM_model_acc1[k])])


    k=k+1

    t_end = time.time()
    print("total time: " + str(t_end-t_start))
    


results = pd.DataFrame(list_params, columns= ["Sample","Method", "Performance", "Sdev"])
ideal_error = np.zeros(5)

print ("end")

wandb.init(project="ib_irm", reinit=True, tags=["colored_mnist"])

plt.figure()
plt.xlabel("Number of samples", fontsize=16)
plt.ylabel("Test error", fontsize=16)
plt.plot(n_tr_list, 1-ERM_model_acc_av, "-r", marker="+", label="ERM")
plt.plot(n_tr_list, 1-IRM_model_acc_av_v, "-b", marker="s",label="IRMv1")
plt.plot(n_tr_list, ideal_error, "-g", marker="x", label="Optimal invariant")
plt.legend(loc="upper left", fontsize=18)
plt.ylim(-0.01,0.8)

wandb.log({"cs_erm_irm_optimal": plt})