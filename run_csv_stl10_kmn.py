import argparse
import os
import numpy as np
import csv

#Jee
import io
import pandas as pd


from google.colab import drive

from datetime import datetime
from keras.models import model_from_json, load_model, save_model
from keras.datasets import cifar100
import tensorflow as tf
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_layerwise_relevances, load_layerwise_relevances
from utils import save_layer_outs, load_layer_outs, get_layer_outs_new
from utils import save_data, load_data, save_quantization, load_quantization
from utils import generate_adversarial, filter_correct_classifications
#from coverages.idc import CombCoverage
from coverages.idc import ImportanceDrivenCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy
from lrp_toolbox.model_io import write, read
from keras.datasets import mnist, cifar10, cifar100

import csv

__version__ = 0.9

import numpy as np
from keras.utils import np_utils

def load_CIFAR100(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=100)
        y_test = np_utils.to_categorical(y_test, num_classes=100)

    return X_train, y_train, X_test, y_test




def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Coverage Analyzer for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")#, required=True)
                        # choices=['lenet1','lenet4', 'lenet5'], required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                         or cifar10).", choices=["mnist","cifar10", "cifar100"])#, required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['idc','nc','kmnc',
                        'nbc','snac','tknc','ssc', 'lsa', 'dsa'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", help="quantization granularity for \
                        combinatorial other_coverage_metrics.", type= int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-KS", "--k_sections", help="number of sections used in \
                        k multisection other_coverage_metrics", type=int)
    parser.add_argument("-KN", "--k_neurons", help="number of neurons used in \
                        top k neuron other_coverage_metrics", type=int)
    parser.add_argument("-RN", "--rel_neurons", help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics", type=int)
    parser.add_argument("-AT", "--act_threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")


    # parse command-line arguments


    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    model_path     = args['model'] if args['model'] else 'neural_networks/LeNet5'
    dataset        = args['dataset'] if args['dataset'] else 'mnist'
    approach       = args['approach'] if args['approach'] else 'cc'
    num_rel_neurons= args['rel_neurons'] if args['rel_neurons'] else 10
    act_threshold  = args['act_threshold'] if args['act_threshold'] else 0
    top_k          = args['k_neurons'] if args['k_neurons'] else 3
    k_sect         = args['k_sections'] if args['k_sections'] else 1000
    selected_class = args['class'] if not args['class']==None else -1 #ALL CLASSES
    repeat         = args['repeat'] if args['repeat'] else 1
    logfile_name   = args['logfile'] if args['logfile'] else 'result.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3
    adv_type      = args['advtype'] if args['advtype'] else 'fgsm'

    logfile = open(logfile_name, 'a')


    ####################
    # 0) Load data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
        img_rows, img_cols = 28, 28

    elif dataset == 'cifar100':
         X_train, Y_train, X_test, Y_test = load_CIFAR100()
         img_rows, img_cols = 32, 32
  

    else:
         #X_train, Y_train, X_test, Y_test = load_CIFAR()
         #X_train, Y_train, X_test, Y_test = load_data("cifar_test_batch0.bin")
         img_rows, img_cols = 96, 96

         #train_path = os.path.join(path='/content/drive/My Drive/deep/deepimportance/sample_data/test_batch_snow_cifar10' +'.bin')
         train_path =('/content/drive/My Drive/deep/deepimportance/sample_data/stl10_brightless_-120.csv')
         train_path2=('/content/drive/My Drive/deep/deepimportance/sample_data/stl10_train200.csv')



         print (train_path)
         
        
         data = np.genfromtxt(train_path, delimiter=',', dtype=None, encoding='UTF-8')

         data2 = np.genfromtxt(train_path2, delimiter=',', dtype=None, encoding='UTF-8')
         print(len(data))

         #csv_
         #with open(train_path, 'r') as f:
         #  reader = csv.reader(f, delimiter = ',')
         #  for txt in reader:
         #    print(txt)

         #file1 = open(train_path, 'r' ,delimiter=',')

         #data = file1.read()

        

         #file1.close()

         X_test = np.zeros((100,96,96,3))  #sample data_set   cifar10, cifar100 : (1000, 32, 32, 3)  mnist : (1000 , 28, 28, 1)  STL10 : (100, 96, 96, 3)

         X_train = np.zeros((200,96,96,3))  #sample data_set   cifar10, cifar100 : (1000, 32, 32, 3)  mnist : (1000 , 28, 28, 1)  STL10 : (100, 96, 96, 3) 

    
         data_num = -1
         data_width = 0 # data width 
         data_height = 0 # data height 
         data_channel = 0 # data channel 


         for k in range(0, 2764900):
           if k % 27649==0:   # 3073 = 32* 32 * 3 +1  / 28*28 +1 = 785
             data_num = data_num + 1
             data_width = 0
             data_height = 0
             data_channel = 0


           elif data_width <96:  # width 32
             #print('test')
             #print(data[0])
             #print(data[1])
             #print(data[2])
             #print(data[3])
             X_test[data_num][data_width][data_height][data_channel] = data[k]
            
             data_width = data_width + 1


           elif data_height == 95 and data_width ==96:  # height 32  channel 
             data_width = 0
             data_height = 0
             data_channel = data_channel + 1
             X_test[data_num][data_width][data_height][data_channel] = data[k]
             data_width = 1

           elif data_width ==96: # width 32 height 
             data_width = 0
             data_height= data_height+1
             X_test[data_num][data_width][data_height][data_channel] = data[k]
             data_width = 1


    
         data_num = -1
         data_width = 0 # data width 
         data_height = 0 # data height 
         data_channel = 0 # data channel 


         for k in range(0, 5529800):  #8294700
           if k % 27649==0:   # 3073 = 32* 32 * 3 +1  / 28*28 +1 = 785
             data_num = data_num + 1
             data_width = 0
             data_height = 0
             data_channel = 0


           elif data_width <96:  # width 32
             #print('test')
             #print(data[0])
             #print(data[1])
             #print(data[2])
             #print(data[3])
             X_train[data_num][data_width][data_height][data_channel] = data2[k]
            
             data_width = data_width + 1


           elif data_height == 95 and data_width ==96:  # height 32  channel 
             data_width = 0
             data_height = 0
             data_channel = data_channel + 1
             X_train[data_num][data_width][data_height][data_channel] = data2[k]
             data_width = 1

           elif data_width ==96: # width 32 height 
             data_width = 0
             data_height= data_height+1
             X_train[data_num][data_width][data_height][data_channel] = data2[k]
             data_width = 1


    if not selected_class == -1:
        X_train, Y_train = filter_val_set(selected_class, X_train, Y_train) #Get training input for selected_class
        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test) #Get testing input for selected_class



    ####################
    # 1) Setup the model
    #model_name = model_path.split('/')[-1]

    #model_name = 'content/drive/My\ Drive/deep/deepimportance/neural_networks/LeNet5'
    model_name = 'content/drive/My\ Drive/deep/deepimportance/neural_networks/cifar100_resnet_32_model'


    try:
        json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        # Compile the model before using
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    except:
        model = load_model(model_path + '.h5', custom_objects=None,compile=False)
        #model = load_model(model_path + '.h5')
        # model = tf.keras.models.load_model(
        #   model_path + '.h5',
        #   custom_objects=None,
        #   compile=False
        # )
        

    # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    #Investigate the penultimate layer
    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = [0] #SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print(skip_layers)

    ####################
    # 3) Analyze Coverages
    if approach == 'nc':
    
        nc = NeuronCoverage(model, threshold=.5, skip_layers = skip_layers) #SKIP ONLY INPUT AND FLATTEN LAYERS
        coverage, _, _, _, _ = nc.test(X_test)
        print("Your test set's coverage is: ", coverage)

        nc.set_measure_state(nc.get_measure_state())

    elif approach == 'idc':
        X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                           X_train,
                                                                           Y_train)
        #CombCoverage -> ImportanceDrivenCoverage
        cc = ImportanceDrivenCoverage(model, model_name, num_rel_neurons, selected_class,
                          subject_layer, X_train_corr, Y_train_corr)
                          #,quantization_granularity)

        coverage, covered_combinations = cc.test(X_test)
        print("Your test set's coverage is: ", coverage)
        print("Number of covered combinations: ", len(covered_combinations))

        cc.set_measure_state(covered_combinations)

    elif approach == 'kmnc' or approach == 'nbc' or approach == 'snac':

        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }

        dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
        score = dg.test(X_test)
        #print("Your test set's score is: ", score)
        #print("Your test set's score is: ", score[0],score[2],score[3])
        print("Your test set's score is: ", score[0])
        print("Your test set's score is: ", score[2])
        print("Your test set's score is: ", score[3])
        #model1 = load_model('/content/drive/My\ Drive/deep/deepimportance/neural_networks/cifar100_resnet_32_model.h5')
        #model = load_model(model_path + '.h5' ,custom_objects=None,compile=False)
        

        #model.compile(loss='categorical_crossentropy',
        #              optimizer='adam',
        #              metrics=['accuracy'])
                      
        #acc = model.evaluate(X_test, Y_test, verbose = 0)
        #acc =  model.fit(X_test, Y_test)
        
        #print("Your test set's accuracy is: ", acc[1]*100)

    elif approach == 'tknc':

        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)
        orig_coverage, _, _, _, _, orig_incrs = dg.test(X_test)
        _, orig_acc =  model.evaluate(X_test, Y_test)
        

    elif approach == 'ssc':

        ss = SSCover(model, skip_layers=non_trainable_layers)
        score = ss.test(X_test)

        print("Your test set's coverage is: ", score)

    elif approach == 'lsa' or approach == 'dsa':
        upper_bound = 2000

        layer_names = [model.layers[-3].name]

        #for lyr in model.layers:
        #    layer_names.append(lyr.name)

        sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset)
        sa.test(X_test, approach)

    logfile.close()

