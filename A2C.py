#Utils
import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

#TextWorld
import textworld
import textworld.gym
from textworld import EnvInfos

#Torch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from rnn import MDNRNN, default_hps, rnn_output,rnn_next_state
import data_handler as dh

from use import USE

#Modes for the RNN output
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)
action_dic = { 0:'look', 1:'inventory', 2:'go north', 3:'go east', 4:'go west', 5:'go south', 6:'take coin', 7:'examine coin' }
df = pd.read_pickle("Datasets/simple_train_data.pkl")


print(device)

class CommandScorer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()

        torch.manual_seed(42)  # For reproducibility
        #self.embedding    = nn.Embedding(input_size, hidden_size)
        #self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
        #self.cmd_encoder_gru  = nn.GRU(hidden_size, hidden_size)
        #self.state_gru    = nn.GRU(hidden_size, hidden_size)

        #USE for embeddings
        self.embedding = USE() 

        #Imagination
        self.hps = default_hps()
        self.RNN = MDNRNN(self.hps)
        self.RNN.load_json("tf_rnn/simple_game_5_cost2.json")
        self.rnn_state = self.RNN.sess.run(self.RNN.initial_state)

        ###################################################
        self.hidden_size  = hidden_size #Ideally 512
        
        #It swaps to rnn last state
        #self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        #intermediary linear layer to avoid an abrupt drop from 512 to 1
        self.critic_h1    = nn.Linear(1024, hidden_size)
        self.critic_h2    = nn.Linear(hidden_size, hidden_size//2)
        self.critic       = nn.Linear(1024, 1)
        
        self.att_cmd      = nn.Linear(hidden_size * 3, 1)
    

    def forward(self, obs, commands, **kwargs):

        #input_length = obs.size(0)
        #batch_size = obs.size(1)
        #nb_cmds = commands.size(1)
        nb_cmds = len(commands)
        #embedded = self.embedding(obs) Change the embedding to USE
        #TF
        embedded = self.embedding.embed(obs)
        embedded = np.reshape(embedded,512)
    
        #Use the immagination
        #TF
        memory = rnn_output(self.rnn_state, embedded, MODE_ZH) #1024 Includes the embeddings

        # Wee no longer need the GRUs
        #encoder_output, encoder_hidden = self.encoder_gru(embedded)
        #state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)

        memory = torch.from_numpy(memory)
        #h1 = self.critic_h1(memory) #Can be changed to use the embedding
        #h2 = self.critic_h2(h1)
        value = self.critic(memory)

        # Attention network over the commands.
        #cmds_embedding = self.embedding.forward(commands)
        #_, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding)  # 1 x cmds x hidden

        #use USE to embed the  commands.
        cmds_embedding = self.embedding.embed_sentences(commands) #List of embeddings
        command_list =  []
        for i in range(len(cmds_embedding)):
            command_list.append(np.reshape(cmds_embedding[i],512))
        command_list = torch.from_numpy(np.array(command_list))
        # Same observed state for all commands.
        cmd_selector_input = torch.stack([memory] * nb_cmds, 0)  # 1 x batch x cmds x hidden
        
        # Same command choices for the whole batch.
        #cmds_encoding_last_states = torch.stack([command_list] * 1, 0)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        #print(cmd_selector_input.shape)
        #print(command_list.shape)
        cmd_selector_input = torch.cat([cmd_selector_input, command_list], dim=-1)

        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds
        #print(scores.shape)
        probs = F.softmax(scores, dim=0)  # 1 x Batch x cmds
        #print(probs)
        index = probs.multinomial(num_samples=1).unsqueeze(0) # 1 x batch x indx
        return scores, index, value

    #def reset_hidden(self, batch_size):
    #    self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)






###################################################################################################################################################

class NeuralAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 1000
    DATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = { 0:'look', 1:'inventory', 2:'go north', 3:'go east', 4:'go west', 5:'go south', 6:'take coin', 7:'examine coin' }
        actions = dh.get_unique_actions(df)
        self.word2id = dh.action_to_num(actions) #{ 'look':0, 'inventory':1, 'go north':2, 'go east':3, 'go west':4, 'go south':5, 'take coin':6, 'examine coin':7 }
        self.UPDATE_FREQUENCY = 2

        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=512) #The input size might be removed since it is not used
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)
        
        self.mode = "test"
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        #self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def test(self):
        self.mode = "test"
        #self.model.reset_hidden(1)
        
    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)
    
    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]
            
            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            
        return self.word2id[word]
            
    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor
      
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:
        
        #We dont need to do word2vec
        
        # Build agent's observation: feedback + look + inventory.
        #input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        
        # Tokenize and pad the input and the commands to chose from.
        #input_tensor = self._process([input_])
        #commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        obs = obs.replace("\n","")
        outputs, indexes, values = self.model(obs, infos["admissible_commands"])
        action = infos["admissible_commands"][indexes[0]]
        
        obs_emb = self.model.embedding.embed(obs)
        obs_emb = np.reshape(obs_emb, 512)

        act = np.zeros(454)
        index = int(indexes[0][0])
        selected_cmd = infos["admissible_commands"][index]
        
        rnn_act = self.word2id[selected_cmd]
        act[rnn_act] = 1

        self.model.rnn_state = rnn_next_state(self.model.RNN, obs_emb , act, self.model.rnn_state)

        if self.mode == "test":
            #if done:
                #self.model.reset_hidden(1)
            return action
        
        self.no_train_step += 1
        
        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100
  
            self.transitions[-1][0] = reward  # Update reward information.
        
        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)
            
            loss = 0
            #print(self.transitions)
            
            
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition
                
                advantage        = advantage.detach() # Block gradients flow here.
                probs            = F.softmax(outputs_, dim=0)
                log_probs        = torch.log(probs)
                log_action_probs = log_probs.gather(0, indexes_[0])
                policy_loss      = (-log_action_probs * advantage).sum()
                value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                entropy     = (-probs * log_probs).sum()
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy
                
                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())
            
            if self.no_train_step % 10 == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {}".format(len(self.word2id))
                print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
            
            loss = torch.from_numpy(np.array(float(loss)))
            loss = Variable(loss.cuda(), requires_grad = True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            self.transitions = []
            #self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call
        
        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
        
        return action