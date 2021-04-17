import pandas as pd
import numpy as np
import os
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
import random
import time
import data_handler as dh
from scipy.stats import pearsonr 
import matplotlib.pyplot as plt

#from vae.vae import ConvVAE, reset_graph
#import use

#The instance of the actual rnn in trainning
seq = 5
#rnn_actual_cost = "rnn_con_similarity_",str(seq),".json"
rnn_actual_cost = "simple_game_5_cost1024.json"
rnn_actual_simm = "simple_game_5_simm1024.json"

from rnn import HyperParams, MDNRNN, rnn_next_state, rnn_init_state, get_pi_idx, sample_sequence

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

# 8 actions for coin collector
# 454 actions for simple game
def default_hps():
  return HyperParams(num_steps= 2000,
                     max_seq_len=seq,
                     input_seq_width=512+454,    # width of our data (512 + 437 actions)
                     output_seq_width=512,    # width of our data is 32
                     rnn_size=1024,    # number of rnn cells
                     batch_size=30,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.000001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=1,
                     recurrent_dropout_prob=0.60,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)


hps = default_hps()
batch_size = hps.batch_size

#Training
data = pd.read_pickle("./Datasets/simple_train_preprocessed.pkl")
data = data.values



#Testing
test_data = pd.read_pickle("./Datasets/simple_test_preprocessed.pkl")
test_data = test_data.values

N_data = data.shape[0]



rnn = MDNRNN(hps)

best_cost = 100

# train loop:


#EMPIEZA EL ENTRENAMIENTO
def train():
  
  coss_simmilarity = 0
  coss_simmilarity_list_test = []
  coss_simmilarity_list_train = []
  start = time.time()

  train_cost_list = []
  test_cost_list = []

  best_cost = 100
  best_simm = 0
  
  curr_learning_rate = hps.learning_rate

  for local_step in range(hps.num_steps):

    step = rnn.sess.run(rnn.global_step)
    
    #curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
    if local_step % 200 == 0 and local_step != 0 and curr_learning_rate > hps.min_learning_rate:
         curr_learning_rate *= 0.9
    
    '''
    print(hps.min_learning_rate)
    print((hps.learning_rate-hps.min_learning_rate))
    print((hps.decay_rate) ** step)
    print(curr_learning_rate)
    '''
    inputs , outputs = dh.create_batch(data, 30, seq)
    
    feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
    (train_cost, state, train_step,_) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
    
    train_cost_list.append(train_cost)
    end = time.time()
    time_taken = end-start
    start = time.time()
    
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)
    if( local_step % 50 == 0 ):
      rnn.is_training = 0
      valid_inputs , valid_outputs = dh.create_batch(test_data, 30, seq)
      feed = {rnn.input_x: valid_inputs, rnn.output_x: valid_outputs}
      (valid_cost, valid_state) = rnn.sess.run([rnn.cost, rnn.final_state], feed)
      #test_cost_list.append(test_cost)
      coss_sim = validate(rnn)
      coss_simmilarity_list_test.append(coss_sim)
      if coss_sim > best_simm:
        rnn.save_json(os.path.join(model_save_path, rnn_actual_simm))
        best_simm = coss_sim 
        
      coss_sim = train_sim(rnn)
      coss_simmilarity_list_train.append(coss_sim)

      if valid_cost < best_cost:
          
        # save the model (don't bother with tf checkpoints json all the way ...)
        rnn.save_json(os.path.join(model_save_path, rnn_actual_cost))
        best_cost = valid_cost 

      rnn.is_training = 1
    test_cost_list.append(valid_cost)
  plt.plot(train_cost_list)
  plt.plot(test_cost_list)
  plt.show()

  plt.plot(coss_simmilarity_list_test)
  plt.plot(coss_simmilarity_list_train)
  plt.show()
  
#TERMINA EL ENTRENAMIENTO

#VALIDACION
def validate(rnn):
    
  rnn.is_training = False
  #Valid
  valid_data = pd.read_pickle("./Datasets/simple_valid_preprocessed.pkl")
  valid_data = valid_data.values[:3000]
  valid_data = dh.create_val_data(valid_data)

  OUTWIDTH = hps.output_seq_width
  INWIDTH = hps.input_seq_width

  #rnn.load_json(jsonfile=rnn_actual_cost)

  prev_x = np.zeros((1, 1, OUTWIDTH))
  prev_state = rnn.sess.run(rnn.initial_state)
  lista = []

  for i in range(len(valid_data)-1):
    
    input_x = valid_data[i]
    feed = {rnn.input_x: input_x, rnn.initial_state[0] : prev_state[0], rnn.initial_state[1] : prev_state[1] }
    [logmix, mean, logstd, next_state] = rnn.sess.run([rnn.out_logmix, rnn.out_mean, rnn.out_logstd, rnn.final_state], feed)

    logmix2 = np.copy(logmix)/1.0

    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    #print(logmix2.shape)
    #logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

    mixture_idx = np.zeros(OUTWIDTH)
    chosen_mean = np.zeros(OUTWIDTH)
    chosen_logstd = np.zeros(OUTWIDTH)


    for j in range(OUTWIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(1.0)
    next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    prev_x[0][0] = next_x
    prev_state = next_state


    #print(next_x.shape)
    #print(valid_data[i+1].shape)
    #print(valid_data[i+1][0][0].shape)
    corr = np.inner(next_x, valid_data[i+1][0][0][:512])
    lista.append(corr)

  k = len(lista)
  suma = 0
  for i in range(k):
    suma += lista[i]#[1]
  print("Validation cossine simmilarity: ",(suma/k))
  rnn.is_training = True
  return suma/k


#SAMPLEO

def sample(rnn):


  rnn.is_training = False
  OUTWIDTH = hps.output_seq_width
  INWIDTH = hps.input_seq_width

  #rnn.load_json(jsonfile=rnn_actual_cost)

  prev_x = np.zeros((1, 1, OUTWIDTH))

  

  inputs , outputs = dh.create_batch(test_data, 300, seq)
  lista = []

  
  prev_state = rnn.sess.run(rnn.initial_state)
  for i in range(5):

    

    input_x = inputs[0,i,:]
    test = outputs[0,i,:]
    input_x = np.expand_dims(input_x,0)
    input_x = np.expand_dims(input_x,0)

    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    
    feed = {rnn.input_x: input_x, rnn.initial_state[0] : prev_state[0], rnn.initial_state[1] : prev_state[1] }
    [logmix, mean, logstd, next_state] = rnn.sess.run([rnn.out_logmix, rnn.out_mean, rnn.out_logstd, rnn.final_state], feed)

    logmix2 = np.copy(logmix)/1.0

    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    #print(logmix2.shape)
    #logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

    mixture_idx = np.zeros(OUTWIDTH)
    chosen_mean = np.zeros(OUTWIDTH)
    chosen_logstd = np.zeros(OUTWIDTH)


    for j in range(OUTWIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(1.0)
    next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    prev_x[0][0] = next_x
    prev_state = next_state


    
    corr = np.inner(next_x, test)
    lista.append(corr)

  k = len(lista)
  suma = 0
  for i in range(k):
    suma += lista[i]#[1]
  rnn.is_training = True
  return suma/k


def train_sim(rnn):

  rnn.is_training = False
  data = pd.read_pickle("./Datasets/simple_train_preprocessed.pkl")
  data = data.values[:3000]
  data = dh.create_val_data(data)

  OUTWIDTH = hps.output_seq_width
  INWIDTH = hps.input_seq_width

  #rnn.load_json(jsonfile=rnn_actual_cost)

  prev_x = np.zeros((1, 1, OUTWIDTH))
  prev_state = rnn.sess.run(rnn.initial_state)
      
  lista = []

  for i in range(len(data)-1):
    
    input_x = data[i]
    #print("AAAAAAAAAAAAAAAAAAAAAAAAA",input_x.shape)
    feed = {rnn.input_x: input_x, rnn.initial_state[0] : prev_state[0], rnn.initial_state[1] : prev_state[1] }
    [logmix, mean, logstd, next_state] = rnn.sess.run([rnn.out_logmix, rnn.out_mean, rnn.out_logstd, rnn.final_state], feed)

    logmix2 = np.copy(logmix)/1.0

    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    #print(logmix2.shape)
    #logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

    mixture_idx = np.zeros(OUTWIDTH)
    chosen_mean = np.zeros(OUTWIDTH)
    chosen_logstd = np.zeros(OUTWIDTH)


    for j in range(OUTWIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(1.0)
    next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    prev_x[0][0] = next_x
    prev_state = next_state


    #print(next_x.shape)
    #print(data[i+1].shape)

    corr = np.inner(next_x, data[i+1][0][0][:512])
    lista.append(corr)

  k = len(lista)
  suma = 0
  for i in range(k):
    suma += lista[i]#[1]
  print("Training cossine simmilarity: ",(suma/k))
  rnn.is_training = True
  return suma/k


def SGD(epochs):
    

    coss_simmilarity = 0
    coss_simmilarity_list_test = []
    coss_simmilarity_list_train = []
    start_time = time.time()

    train_cost_list = []
    test_cost_list = []

    best_cost = 100
    best_simm = 0

    
    

    for i in range(epochs):
          
      
      data = dh.SGD_data('./Datasets/train1.pkl',600000)
      #valid_data = dh.SGD_data('./Datasets/valid1.pkl',250000)
      
      #print(start, hps.batch_size)
      print("Getting started with epoch: ",i)
      start = 0

      while start+hps.batch_size < data.shape[0]-1:
            
        print(start)
        
        step = rnn.sess.run(rnn.global_step)   
        
        inputs = data[start:][:hps.batch_size]
        outputs = data[start+1:][:hps.batch_size]
        outputs = outputs[:,:,:512]
        #print(inputs.shape, outputs.shape)
        
        start += hps.batch_size
        #print(start+hps.batch_size)
        
        curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
        
        feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
        (train_cost, state, train_step,_) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
        
        train_cost_list.append(train_cost)
        end_time = time.time()
        time_taken = end_time-start_time
        start_time = time.time()
        
        
        
        
      output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
      print(output_log)
      if( i % 1 == 0 ):
          rnn.is_training = 0

          valid_inputs , valid_outputs = dh.create_batch(test_data, 30, seq)

          feed = {rnn.input_x: valid_inputs, rnn.output_x: valid_outputs}
          (valid_cost, valid_state) = rnn.sess.run([rnn.cost, rnn.final_state], feed)

          test_cost_list.append(valid_cost)

          coss_sim = validate(rnn)
          coss_simmilarity_list_test.append(coss_sim)

          if coss_simmilarity > best_simm:
            rnn.save_json(os.path.join(model_save_path, rnn_actual_simm))
            best_simm = coss_sim 
            
          coss_sim = train_sim(rnn)
          coss_simmilarity_list_train.append(coss_sim)

          if valid_cost < best_cost:
              
            # save the model (don't bother with tf checkpoints json all the way ...)
            rnn.save_json(os.path.join(model_save_path, rnn_actual_cost))
            best_cost = valid_cost 

          rnn.is_training = 1
    plt.plot(train_cost_list)
    plt.plot(test_cost_list)
    plt.show()

    plt.plot(coss_simmilarity_list_test)
    plt.plot(coss_simmilarity_list_train)
    plt.show()      


train()