import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

epochs=[]
for i in range(50):
    epochs.append(i)

#rewards_hopper_3 = pd.read_csv('./files/aviation-noa(3modes)-testing.csv')
rewards_hopper_5 = pd.read_csv('./files/log-5modes-testing.csv')

def plot_reward(epochs, rewards, title, savepath):
    epochs = np.asarray(epochs)
    rewards = np.asarray(rewards)

    epochs = epochs.copy()
    save_path = savepath.format('5')#'./plots/aviation_noa_testing-rewards-modes{}.png'.format('3')
    pred_context = rewards.copy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(epochs, pred_context, c='r', marker='X')
    ax.plot(epochs, pred_context, color='Blue')#, color="cornflowerblue", color="Orange
    plt.xlabel("Episodes")
    #plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1500])
    # naming the y axis
    plt.ylabel("Testing Rewards")
    # giving a title to my graph
    plt.title(title)#Aviation-NOA (3 Modes) - Directed-Info Gail
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

#plot_reward(epochs, rewards_raw_3, 'Hopper-V2 (3 Modes) - Directed-Info Gail', './plots/hopper_testing-rewards-modes{}.png')
plot_reward(epochs, rewards_hopper_5, 'Hopper-V2 (5 Modes) - Directed-Info Gail', './plots/hopper_testing-rewards-modes{}.png')