import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

epochs=[]
for i in range(50):
    epochs.append(i)

rewards = pd.read_csv('./files/aviation-noa(3modes)-testing.csv')
rewards_noa_5 = pd.read_csv('./files/aviation-noa(5modes)-testing.csv')

rewards_metar_3 = pd.read_csv('./files/aviation-metar(3modes)-testing.csv')
rewards_metar_5 = pd.read_csv('./files/aviation-metar(5modes)-testing.csv')

rewards_raw_3 = pd.read_csv('./files/aviation-raw(3modes)-testing.csv')
rewards_raw_5 = pd.read_csv('./files/aviation-raw(5modes)-testing.csv')

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

#plot_reward(epochs, rewards, 'Aviation-NOA (3 Modes) - Directed-Info Gail', '')
#plot_reward(epochs, rewards_noa_5, 'Aviation-NOA (5 Modes) - Directed-Info Gail', '')
#plot_reward(epochs, rewards_metar_3, 'Aviation-METAR (3 Modes) - Directed-Info Gail', './plots/aviation_metar_testing-rewards-modes{}.png')
#plot_reward(epochs, rewards_metar_5, 'Aviation-METAR (5 Modes) - Directed-Info Gail', './plots/aviation_metar_testing-rewards-modes{}.png')
#plot_reward(epochs, rewards_raw_3, 'Aviation-RAW (3 Modes) - Directed-Info Gail', './plots/aviation_raw_testing-rewards-modes{}.png')
plot_reward(epochs, rewards_raw_5, 'Aviation-RAW (5 Modes) - Directed-Info Gail', './plots/aviation_raw_testing-rewards-modes{}.png')