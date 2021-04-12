import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import use
#u = use.USE()

def load_json( jsonfile='train_dataset.json'):
    with open(jsonfile, 'r') as f:
        params = json.load(f)
    return params


def pandas_load(jsonfile='train_dataset.json'):
    df = pd.read_json(jsonfile, lines= True)
    return df

def get_unique_actions(df):
    res = pd.unique(df["act"])
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

#Deprecated
def processed_data(data):
    encodings = []
    actions = []
    df = pd.DataFrame(columns=['obs','n'])
    accions = get_unique_actions(data)
    d = action_to_num(accions)
    j = 0
    N = data.shape[0]
    for i in tqdm (range(N), desc= "Processing..."):
        enc = u.embed(data.values[i][0])
        enc = np.array(enc)
        enc.resize((512,))
        action = d[data.values[i][1]]
        encodings.append(enc)
        actions.append(action)
    df = pd.DataFrame({'obs':encodings,'n':actions})
    p_data =np.array(df)
    with open('p_data.pkl','wb') as f:
        df.to_pickle(f)
    return df

#Deprecated
def processed_test_data():
    train = pandas_load()
    test = pandas_load(jsonfile = 'test_dataset.json')

    train_actions = get_unique_actions(train)
    test_actions = get_unique_actions(test)

    dif = []
    for i in test_actions:
        if i not in train_actions:
            dif.append(i)

    u_actions = get_unique_actions(train)
    d = action_to_num(u_actions)
    N = test.shape[0]
    
    encodings = []
    actions = []
    for i in tqdm (range(N), desc= "Processing..."):
        if test.values[i][1] in u_actions:
            enc = u.embed(test.values[i][0])
            enc = np.array(enc)
            enc.resize((512,))
            action = d[test.values[i][1]]
            encodings.append(enc)
            actions.append(action)
    df = pd.DataFrame({'obs':encodings,'n':actions})
    p_data =np.array(df)
    with open('p_test_data.pkl','wb') as f:
        df.to_pickle(f)
    return df


def create_batch(data, batch_size, look_back):
    indices = np.random.permutation(data.shape[0])[0:batch_size]
    n_seq = data.shape[-1]
    X = np.zeros((batch_size,look_back,512+8))
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
        acc = np.zeros(8)
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


def new_preprocess(df):
    
    accions = get_unique_actions(df)
    d = action_to_num(accions)
    for i in range(df.shape[0]):
        #print(d[df.values[i][1]])
        df.values[i][1] = d[df.values[i][1]]
    df.to_pickle('valid1.pkl')

def create_val_data(data):
    
    final_data = []
    for i in data:
        acc_arr = np.zeros((1,1,8)) 
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

    ohe = np.zeros(8)
    final_dataset = np.zeros((size, 520))

    for i in range(size):
        ohe[ data[i][1] ] = 1
        final_data = np.concatenate( (data[i][0],ohe) )
        ohe = np.zeros(8)
        final_dataset[i] = final_data

    final_dataset= np.reshape(final_dataset,(int(size/5),5,520))
    rnd = np.random.permutation(int(size/5)) 
    final_dataset = final_dataset[rnd]

    return final_dataset

df = pd.read_pickle("./Datasets/train1.pkl")

print(df['act'].unique())