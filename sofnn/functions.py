from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras import backend as K
import numpy as np

def loss_function(y_true, y_pred):
        """
        Custom loss function

        E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}

        Parameters
        ==========
        y_true : np.array
            - true values
        y_pred : np.array
            - predicted values
        """
        return K.sum(1 / 2 * K.square(y_pred - y_true))

def measures(y_test_ori,yhat_classes):
        testy = y_test_ori
        #print("yhat_classes",yhat_classes)
        #print("testy",testy)
        accuracy = accuracy_score(testy, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(testy, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(testy, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy, yhat_classes)
        print('F1 score: %f' % f1)
def pred_binarisation(preds):
        binary=[]
        for i in range(0,len(preds)):
            if(preds[i][0]>=preds[i][1]):
                binary.append(0)
            else:
                binary.append(1)
        return binary


def calculate_num_sets(centers, sigmas):
  mfs=[]
  l=[]
  for i in range(0,len(centers)):
    my_list=[]
    for j in range(0,len(centers[0])):
      my_list.append([centers[i][j],sigmas[i][j]])
    mfs.append(my_list)
  for i in range(0,len(mfs)):
    l.append(len(set(tuple(row) for row in mfs[i])))
  return (l)