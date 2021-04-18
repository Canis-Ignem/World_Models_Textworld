import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import use
u = use.USE()


def get_unique_actions(df):
    res = pd.unique(df["cmd"])
    return res

def action_to_num(df):
    d = {}
    i = 0
    for action in df:
        d[action] = i
        i += 1
    return d

def get_num(d,s):
    return d[s]



def create_batch(data, batch_size, look_back):
    indices = np.random.permutation(data.shape[0])[0:batch_size]
    n_seq = data.shape[-1]
    X = np.zeros((batch_size,look_back,512+454))
    Y = np.zeros((batch_size,look_back,512))
    for i in range(len(indices)):
        create_Data_Multiple(data, batch_size, look_back, indices[i], i, X, Y)
    return X, Y
    
def create_Data_Multiple(data, batch_size, look_back,  index_to_predict, batch_idx, X, Y):
    
    #Y[batch_idx,look_back-1,:] = data[index_to_predict][0]
    i = 0
    inputs = []
    while i < look_back  and index_to_predict -i >0:
        
        obs = data[index_to_predict-1-i,0]
        acc = np.zeros(454)
        acc[data[index_to_predict-i,1]] = 1
        a = np.concatenate((obs,acc), axis=0)
        
        Y[batch_idx,look_back-i-1,:] = data[index_to_predict-i][0]
        #
        #data[index_to_predict][0]
        
        inputs.append(a)
        i += 1
    
        for j in np.arange(look_back):
            for k in range(len(inputs)):
                X[batch_idx,look_back-j-1,:] = inputs[k]
    return X,Y



def clean_obs(df):

    for i in range(df.shape[0]):
        df[i][0] = df.values[i][0].replace("\n"," ")

def new_preprocess(train, test, valid):
    
    actions = get_unique_actions(train)
    d = action_to_num(actions)

    l = []
    print("Processing the trainning data")
    for i in tqdm(range(train.shape[0])):
        
        obs = train.values[i][0].replace("\n"," ")
        emb_obs = u.embed(obs)
        emb_obs = np.reshape(emb_obs, 512) 
        train.values[i][0] = emb_obs
        train.values[i][1] = d[train.values[i][1]]
    

    train.to_pickle('./Datasets/simple_train_preprocessed.pkl')
    print("Processing the testing data")
    for i in tqdm(range(test.shape[0])):
        pair = []
        if test.values[i][1] in d:
            obs = test.values[i][0].replace("\n"," ")
            emb_obs = u.embed(obs)
            emb_obs = np.reshape(emb_obs, 512) 
            pair.append(emb_obs)
            pair.append(d[test.values[i][1]])
            l.append(pair)
       
        '''
        test.values[i][1] = d[test.values[i][1]]
        obs = test.values[i][0].replace("\n"," ")
        emb_obs = u.embed(obs)
        emb_obs = np.reshape(emb_obs, 512) 
        test.values[i][0] = emb_obs
        '''
    test = pd.DataFrame(l, columns={'obs','cmd'})
    test.to_pickle('./Datasets/simple_test_preprocessed.pkl')
    l= []
    print("Processing the validation data")
    for i in tqdm(range(valid.shape[0])):
        pair = []
        if valid.values[i][1] in d:
            obs = valid.values[i][0].replace("\n"," ")
            emb_obs = u.embed(obs)
            emb_obs = np.reshape(emb_obs, 512) 
            pair.append(emb_obs)
            pair.append(d[valid.values[i][1]])
            l.append(pair)

    valid = pd.DataFrame(l, columns={'obs','cmd'})
    valid.to_pickle('./Datasets/simple_valid_preprocessed.pkl')


def create_val_data(data):
    
    final_data = []
    for i in data:
        acc_arr = np.zeros((1,1,454)) 
        obs = i[0]
        obs = np.expand_dims(obs,0)
        obs = np.expand_dims(obs,0)
        acc_arr[0][0][i[1]] = 1
        con = np.concatenate((obs,acc_arr),2)
        final_data.append(con)
    return final_data

def SGD_data(path,size):

    with open(path ,'rb') as f:
        data = pd.read_pickle(f)

    data = data.values
    data = data[:size]

    ohe = np.zeros(454)
    final_dataset = np.zeros((size, 512+454))

    for i in range(size):
        ohe[ data[i][1] ] = 1
        final_data = np.concatenate( (data[i][0],ohe) )
        ohe = np.zeros(454)
        final_dataset[i] = final_data

    final_dataset= np.reshape(final_dataset,(int(size/10),10,512+454))
    rnd = np.random.permutation(int(size/10)) 
    final_dataset = final_dataset[rnd]

    return final_dataset


'''
train = pd.read_pickle("./Datasets/simple_train_data.pkl")
test = pd.read_pickle("./Datasets/simple_test_data.pkl")
valid = pd.read_pickle("./Datasets/simple_valid_data.pkl")

new_preprocess(train, test, valid)
'''