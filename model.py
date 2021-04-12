import numpy as np
import random
import json
import sys
import os
import time
from glob import glob


from rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

from textworld import EnvInfos
import textworld.gym
import gym

from use import USE

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

action_dic = { 0:'look', 1:'inventory', 2:'go north', 3:'go east', 4:'go west', 5:'go south', 6:'take coin', 7:'examine coin' }

#U = USE()

def make_model(load_model=True):
  # can be extended in the future.
  model = Model(load_model=load_model)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self, load_model=True):
    self.use = USE()
    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      #self.use.load_json('vae/vae.json')
      self.rnn.load_json('tf_rnn/rnn_dataset_generado2.json')

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 512
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 600
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 8)
      self.bias_output = np.random.randn(8)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+8)
    else:
      self.weight = np.random.randn(self.input_size, 8)
      self.bias = np.random.randn(8)
      self.param_count = (self.input_size)*3+8

    print("AAAAAAAAAAAAAAAAAAAAA")
    #print(self.weight_hidden)
    #print(self.bias__hidden)
    print(self.bias.shape)
    print(self.weight.shape)
  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    
    enc = self.use.embed(obs)
    #enc = np.array(enc)
    #enc.resize((512,))
    return enc

  def get_action(self, z):
    h = rnn_output(self.state, z, EXP_MODE)

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    
    #action[1] = (action[1]+1.0) / 2.0
    #action[2] = clip(action[2])

    self.state = rnn_next_state(self.rnn, z, action, self.state)
    command = np.where(action == np.amax(action))
    #action = np.sort(action)
    return command[0][0]

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:3]
      self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
    else:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    #vae_params = self.use.get_random_model_params(stdev=stdev)
    #self.use.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

  @property
  def infos_to_request(self) -> EnvInfos:
    return EnvInfos(admissible_commands=True)


def play(agent, path, max_step=100, train_mode = 0, nb_episodes=10, verbose=True):
    

  reward_list = []
  t_list = []
  infos_to_request = EnvInfos(admissible_commands=True)
  
  gamefiles = [path]
  if os.path.isdir(path):
      gamefiles = glob(os.path.join(path, "*.ulx"))
      
  env_id = textworld.gym.register_games(gamefiles,
                                        request_infos=infos_to_request,
                                        max_episode_steps=max_step)
  env = gym.make(env_id)  # Create a Gym environment to play the text game.
  if verbose:
      if os.path.isdir(path):
          print(os.path.dirname(path), end="")
      else:
          print(os.path.basename(path), end="")
      
  # Collect some statistics: nb_steps, final reward.
  avg_moves, avg_scores, avg_norm_scores = [], [], []

  for no_episode in range(nb_episodes):
      
      obs, infos = env.reset()  # Start new episode.

      total_reward = 0.0

      print(obs)
      score = 0
      done = False
      nb_moves = 0
      k = 0
      while not done:
          if k % 100 == 0:
            print("Episodio: ", no_episode)
          
          enc = agent.encode_obs(obs)
          enc = np.array(enc)
          enc.resize((512,))
          #print(obs)
          #print("Las opciones son:" ,infos["admissible_commands"])
          
          k += 1
          command = agent.get_action(enc)
          #a = action_dic[command]
          
          if action_dic[command] in infos["admissible_commands"]:
                action = action_dic[command]
          else:
                action = 'look'

          print(action_dic[command])
          print(infos["admissible_commands"])

          obs, score, done, infos = env.step(action)
          total_reward += score
          nb_moves += 1
          
      reward_list.append(total_reward)
      t_list.append(nb_moves)
      #agent.act(obs, score, done, infos)  # Let the agent know the game is done.
              
      if verbose:
          print(".", end="")
      #avg_moves.append(nb_moves)
      #avg_scores.append(score)
      #avg_norm_scores.append(score / infos["max_score"])

  env.close()
  msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
  if verbose:
      if os.path.isdir(path):
          print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
      else:
          print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
  return reward_list, t_list
agent = Model()
play(agent,"./games",train_mode = False, max_step=100, nb_episodes=1, verbose=True)











'''
def simulate(model, train_mode=False, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  max_episode_length = 1000
  recording_mode = False
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()

    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    filename = "record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]

    for t in range(max_episode_length):

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)

      obs, reward, done, info = model.env.step(action)

      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward

      recording_reward.append(reward)

      total_reward += reward

      if done:
        break

    #for recording:
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)


  return reward_list, t_list

'''