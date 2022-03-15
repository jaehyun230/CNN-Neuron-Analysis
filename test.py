import argparse

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
    dataset        = args['dataset'] if args['dataset'] else 'cifar10'  #mnist_original
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


    ####################
    # 0) Load data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
        img_rows, img_cols = 28, 28

    elif dataset == 'cifar100':
         X_train, Y_train, X_test, Y_test = load_CIFAR100()
         img_rows, img_cols = 32, 32

    elif dataset == 'my_data':
         X_train, Y_train, X_test, Y_test = load_CIFAR100()
         img_rows, img_cols = 32, 32
  

    else:
         X_train, Y_train, X_test, Y_test = load_CIFAR()
         #X_train, Y_train, X_test, Y_test = load_data("cifar_test_batch0.bin")
         img_rows, img_cols = 32, 32

    if not selected_class == -1:
        X_train, Y_train = filter_val_set(selected_class, X_train, Y_train) #Get training input for selected_class
        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test) #Get testing input for selected_class
    

    #dataset== cifar100
    print (X_train)
    print(X_train.shape)
    print (X_train[49999][31][31][2])
    print (X_train[49999][31][30][2])
    print (X_train[49999][31][29][2])
    
    #X_train[1][27][27][0]= 1
    #print(X_train[1][27][27][0])
    #print(type(X_train))
    #print (len(X_train))
    #print (X_test)
    #print (Y_train)
    #print (Y_test)

  




