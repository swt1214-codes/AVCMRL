
from __future__ import print_function
import pickle as pickle
import os
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data_path =  '../dataset_process/'
dump_path = '../dataset_pkl/'


os.makedirs(dump_path,exist_ok=True)

def read_RBPpred():

    view1_train = scio.loadmat(data_path + 'RBPpred_3viewtrain')['view1']
    view2_train = scio.loadmat(data_path + 'RBPpred_3viewtrain')['view2']
    view3_train = scio.loadmat(data_path + 'RBPpred_3viewtrain')['view3']
    label_train = scio.loadmat(data_path + 'RBPpred_3viewtrain')['label']

    view1_val = scio.loadmat(data_path + 'RBPpred_valcheck')['view1']
    view2_val = scio.loadmat(data_path + 'RBPpred_valcheck')['view2']
    view3_val = scio.loadmat(data_path + 'RBPpred_valcheck')['view3']
    label_val = scio.loadmat(data_path + 'RBPpred_valcheck')['label']

    view1_test = scio.loadmat(data_path + 'RBPpred_3viewtest')['view1']
    view2_test = scio.loadmat(data_path + 'RBPpred_3viewtest')['view2']
    view3_test = scio.loadmat(data_path + 'RBPpred_3viewtest')['view3']
    label_test = scio.loadmat(data_path + 'RBPpred_3viewtest')['label']


    return view1_train, view2_train, view3_train, label_train, \
           view1_val, view2_val,view3_val,label_val, \
           view1_test, view2_test, view3_test, label_test



def write_RBPpred():
    (view1_train, view2_train, view3_train, label_train, view1_val, view2_val, view3_val, label_val, view1_test, view2_test, view3_test, label_test) = read_RBPpred()

    with open(os.path.join(dump_path, 'RBP3viewtrain.pkl'), 'wb') as f_train:
        pickle.dump((view1_train, view2_train, view3_train, label_train), f_train, -1)

    with open(os.path.join(dump_path, 'RBP3viewtest.pkl'), 'wb') as f_test:
        pickle.dump((view1_test, view2_test, view3_test, label_test), f_test, -1)

    with open(os.path.join(dump_path, 'RBP3valcheck.pkl'), 'wb') as f_val:
        pickle.dump((view1_val, view2_val, view3_val, label_val), f_val, -1)

    print("finished!============================")



# def test_write():
#     write()
#     print("\n\n\nRead data ===========================================================================")
#     with open(dump_path + 'webtest.pkl', 'rb') as fp:
#         train_page_data, train_link_data, train_labels = pickle.load(fp)
#         print("test size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
#     with open(dump_path + 'webtrain.pkl', 'rb') as fp:
#         train_page_data, train_link_data, train_labels = pickle.load(fp)
#         print("train size = ", len(train_page_data), train_page_data.shape, train_link_data.shape)




if __name__ == "__main__":
    # test_write()
    write_RBPpred()

