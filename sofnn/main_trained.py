import pandas as pd
from SelfOrganizer import *
from FuzzyNetwork import *
import numpy as np
import argparse
import keras
from layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer
from functions import *


def main(args):
    
    '''
    load dataset
    '''
    if args.dataset == 'wbc':
        data_test=pd.read_csv('splitted_data/test_wisc.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'spectf':
        data_test=pd.read_csv('splitted_data/test_spectf.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'parkinson':
        data_test=pd.read_csv('splitted_data/test_park.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'diabets':
        data_test=pd.read_csv('splitted_data/test_diabets.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    else:
        print('dataset does not exist')
        assert False

    X_test = x_test.to_numpy()
    Y_test = y_test.to_numpy()


    # file_name = "trained_models/"+args.dataset+"_model.pkl"
    # filehandler = open(file_name, 'rb')
    # trained_model = pickle.load(filehandler)
    # print("new: ",trained_model)

    # filehandler = open("wisc_fuzz.pkl", 'wb')
    # fuzz = pickle.load(filehandler)
    # print("Fuzzy layer: ",fuzz)

    reconstructed_model = keras.models.load_model("trained_models/"+args.dataset+"_model.h5", custom_objects={'FuzzyLayer': FuzzyLayer, 'NormalizedLayer':NormalizedLayer,'WeightedLayer':WeightedLayer, 'OutputLayer':OutputLayer,'loss_function':loss_function})
    print("reconstructed_model: ",reconstructed_model)
    preds = reconstructed_model.predict(X_test)
    yhat_classes = pred_binarisation(preds)
    measures(Y_test,yhat_classes)
    print("Model Summary",reconstructed_model.summary())

    w=reconstructed_model.get_weights()
    # print("length all weights",len(w))
    centers = w[0]
    sigmas = w[1]
    # print("Centers: ",centers)
    # print("sigmas:", sigmas)
    # print("Number of centers",len(centers[0]))
    # print("Number of sigmas",len(sigmas[1]))

    num_uniq_sets = calculate_num_sets(centers, sigmas)
    print("Number of unique sets: ",num_uniq_sets)
    print("Mean MFs: ", np.mean(num_uniq_sets))


    print("Y_test: ", Y_test)
    print("Y_Predicted: ",yhat_classes)

  
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sofnn')
    parser.add_argument('--dataset', default='wbc', type=str, help='dataset to load')
        
    args = parser.parse_args()
    main(args)

