import pandas as pd
from sklearn.model_selection import train_test_split
from SelfOrganizer import *
from FuzzyNetwork import *
import numpy as np
import datetime
import argparse



def main(args):
    
    '''
    load dataset
    '''
    if args.dataset == 'wbc':
        data_train=pd.read_csv('splitted_data/train_wisc.csv')
        y_train=data_train.c
        x_train=data_train.drop('c',axis=1)
        data_test=pd.read_csv('splitted_data/test_wisc.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'spectf':
        data_train=pd.read_csv('splitted_data/train_spectf.csv')
        y_train=data_train.c
        x_train=data_train.drop('c',axis=1)
        data_test=pd.read_csv('splitted_data/test_spectf.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'parkinson':
        data_train=pd.read_csv('splitted_data/train_park.csv')
        y_train=data_train.c
        x_train=data_train.drop('c',axis=1)
        data_test=pd.read_csv('splitted_data/test_park.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    elif args.dataset == 'diabets':
        data_train=pd.read_csv('splitted_data/train_diabets.csv')
        y_train=data_train.c
        x_train=data_train.drop('c',axis=1)
        data_test=pd.read_csv('splitted_data/test_diabets.csv')
        y_test=data_test.c
        x_test=data_test.drop('c',axis=1)
    else:
        print('dataset does not exist')
        assert False
#    data=pd.read_csv('wisconsin.csv')
#    data=pd.read_csv('spectf.csv')
#    data.drop('id_number', axis='columns', inplace=True)

#    y=data.c
#    x=data.drop('c',axis=1)
#    y=data.diagnosis
#    x=data.drop('diagnosis',axis=1)


    #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

    X_train = x_train.to_numpy()
    X_test = x_test.to_numpy()
    Y_train = y_train.to_numpy()
    Y_test = y_test.to_numpy()

    
    
    old_time = datetime.datetime.now()
    SO = SelfOrganizer()
    SO.build_network(X_train, X_test, Y_train, Y_test)
    SO.compile_model()
    SO.self_organize()

    new_time = datetime.datetime.now()
    timing = new_time-old_time
    print("Timing: ",timing.total_seconds())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    print("measures: ",SO.network.measures())

    
    print("Summary after self_organize",SO.model.summary())

    # w=SO.model.get_weights()
    # print("all weights",w)
    # centers = w[0]
    # sigmas = w[1]
    # print("Centers: ",centers)
    # print("sigmas:", sigmas)
    # print("Number of centers",len(centers[0]))
    # print("Number of sigmas",len(sigmas[1]))


    


    #### save model
    # final_model = SO.model
    # final_model.save("trained_models/"+args.dataset+"_model.h5")
    

  
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sofnn')
    parser.add_argument('--dataset', default='wbc', type=str, help='dataset to load')
        
    args = parser.parse_args()
    main(args)
