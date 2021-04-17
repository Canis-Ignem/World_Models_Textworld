import os
from glob import glob
import gym
import textworld.gym
from matplotlib import pyplot as plt
import numpy as np
import A2C

agent = A2C.NeuralAgent()

def play(agent, path, max_step=200, nb_episodes=5, verbose=True):
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
    wins = 0
    
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

        score = 0
        done = False
        nb_moves = 0
        while not done:
            command = agent.act(obs, score, done, infos)
            obs, score, done, infos = env.step(command)
            if score > 0:
                wins +=1
            nb_moves += 1
            #print(nb_moves)
        
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.
                
        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    env.close()
    msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    '''
    if verbose:
        if os.path.isdir(path):
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
    '''
    print(wins/nb_episodes)

agent.train()
play(agent,"games/tw-coin_collector-qDiPhg-house-mo-KEe7S5rjtKb5sO8Y.ulx", nb_episodes=50)

for stat in agent.stats["mean"]:
    plt.plot(stat)
    plt.show()