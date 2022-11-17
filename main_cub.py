# -*- coding: utf-8 -*-

# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Torch-related
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader


# Local imports
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz
from SENN.arglist import get_senn_parser


from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.template_model import inception_v3

from models import GSENN
from conceptizers_CUB import image_fcc_conceptizer, image_cnn_conceptizer
from parametrizers import image_parametrizer, dfc_parametrizer
from aggregators_CUB import additive_scalar_aggregator, CBM_aggregator
from trainers_CUB import GradPenaltyTrainer


# This function does not modification
def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


"""
Main function:
load data, set models, train and test, and save results.
After ending this function, you can see <./out/cub/*> directory to check outputs.

Inputs:
    None
Returns:
    None

Inputs loaded in this function:
    ./data/CUB_200_2011: images of CUB_200_2011
    ./data/CUB_200_2011/train.pkl, val.pkl, test.pkl: train, val, test samples splited by CBM's authors
    ./models/inception_v3_google-1a9a5a14.pth: inception v3 network pretrained by ImageNet, using CBM

Outputs made in this function:
    *.pkl: model
    grad*/training_losses.pdf: loss figure
    grad*/concept_grid.pdf: images which maximize and minimize each unit in the concept layer
    grad*/test_results_of_cub.csv: predicted and correct labels, predicted and correct concepts, coefficient of each concept 
"""
def main():
    # get hyperparameters
    args = parse_args()
    # the number of task class
    args.nclasses = 200
    args.theta_dim = args.nclasses

    # set which GPU uses
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    # load dataset
    # if you get the final performance, please comment out and retrain...
    train_data_path = "./data/CUB_200_2011/train.pkl"
    val_data_path = "./data/CUB_200_2011/val.pkl"
    test_data_path = "./data/CUB_200_2011/test.pkl"
    
    # load_data. Detail is CUB/dataset.py, lines 149-. This function is made by CBM's authors
    train_loader = load_data([train_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False)
    valid_loader = load_data([val_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False)
    """
    train_loader = load_data([train_data_path,val_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False)
    valid_loader = load_data([val_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False)
    """
    test_loader = load_data([test_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False)
    
    # get paths (see SENN/utils.py, lines 34-). This function is made by SENN's authors
    model_path, log_path, results_path = generate_dir_names('cub', args)
    
    # initialize the csv file (cleaning before training)
    save_file_name = "%s/test_results_of_cub.csv"%(results_path)
    fp = open(save_file_name,'w')
    fp.close()


    """
    Next, we set four networks, conceptizer, parametrizer, aggregator, and pretrained_model
    Pretrained_model (e(x)): encoder (e) inception v.3 (see ./CUB/template_model.py)
    Conceptizer (c(e(x))): concepts layer (see conceptizer.py)
    Parametrizer (t(e(x))): network to compute parameters to get concepts (see parametrizer.py)
    Aggregator (f(c(e(x)),t(e(x)))): output layer (see aggregators.py)
    
    Note: we use the inception v3 network as the encoder. This model uses the auxiliary output 
    (classification layer is not only the output layer but also the intermidiate layer)
    Thus, we have to set the auxiliary version networks (conceptizers2, parametrizers2, ...)
    """    
    # only "fcc" conceptizer use, otherwise cannot use (not modifile so as to fit this task...)
    if args.h_type == "fcc":
        conceptizer1  = image_fcc_conceptizer(2048, args.nconcepts, args.nconcepts_labeled, args.concept_dim, args.h_sparsity, args.senn)
        conceptizer2  = image_fcc_conceptizer(768, args.nconcepts, args.nconcepts_labeled, args.concept_dim, args.h_sparsity, args.senn)
    elif args.h_type == 'cnn':
        conceptizer  = image_cnn_conceptizer(28*28, args.nconcepts, args.nconcepts_labeled, args.concept_dim, args.h_sparsity)
        print("[ERROR] please use fcc network")
        sys.exit(1)
    else:
        conceptizer  = input_conceptizer()
        args.nconcepts = 28*28 + int(not args.nobias)
        print("[ERROR] please use fcc network")
        sys.exit(1)

    parametrizer1 = dfc_parametrizer(2048,1024,512,256,128,args.nconcepts, args.theta_dim, layers=4)
    buf = 1
    parametrizer2 = dfc_parametrizer(768,512,256,128,buf,args.nconcepts, args.theta_dim,layers=3)

    """
    If you train CBM model, set cbm, <python main_cub.py --cbm>.
    In this case, our model does not use unknown concepts even if you set the number of unknown concepts.
    NOTE: # of unknown concepts = args.nconcepts - args.nconcepts_labeled
    """
    if args.cbm == True:
        aggregator = CBM_aggregator(args.concept_dim, args.nclasses, args.nconcepts_labeled)
    else:
        aggregator = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    # you should set load_model as True. If you set, you can use inception v.3 as the encoder, otherwise end.
    if args.load_model:
        pretrained_model = inception_v3(pretrained=True, freeze=False, num_classes=N_CLASSES, aux_logits=True,
                        n_attributes=args.nconcepts_labeled, bottleneck=True, expand_dim=0,
                        three_class=False)
        pretrained_model = pretrained_model.to(device)
    else:
        print("[ERROR] Please set load_model, <python main_cub.py --load_model>")
        sys.exit(1)

    """
    Function GSENN is in models.py
    model: model using outputs of inception v.3
    model_aux: mdoel using auxiliary output of inception v.3
    """
    model = GSENN(conceptizer1, parametrizer1, aggregator, args.cbm, args.senn) 
    model_aux = GSENN(conceptizer2, parametrizer2, aggregator, args.cbm, args.senn) 
    
    # send models to device you want to use
    model = model.to(device)
    model_aux = model_aux.to(device)
    
    # training all models. This function is in trainers.py    
    trainer = GradPenaltyTrainer(model, model_aux, pretrained_model, args, device)

    # train
    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
    
    # make figures
    trainer.plot_losses(save_path=results_path)

    # evaluation by test dataset
    trainer.test_and_save(test_loader, save_file_name, fold = 'test')
    
    # send model result to cpu
    model.eval().to("cpu")
    pretrained_model = pretrained_model.to("cpu")
    

    """
    This function is in SENN/utils.py (lines 591-). 
    This function makes figures "grad*/concept_grid.pdf", which represents the maximize and minimize each unit in the concept layer
    """
    #concept_grid(model, pretrained_model, test_loader, top_k = 10, device="cpu", save_path = results_path + '/concept_grid.pdf')

    
if __name__ == '__main__':
    main()
