myconfig = {
    'output_dir': '.',
    'log_dir': './logs',
    'input_dir': '.',
    'exp': 0,
    'path_size': 1000,
    'mini_batch_size': 50000,
    'mini_batches': 1500,
    'gamma': 0.995,
    'lamda': 0.97,
    'max_kl': 0.9,#0.01
    'logstd': 0.6,
    'critic_epochs': 100,
    #'critic_alpha': 0.0001,
    'critic_alpha': 0.001,
    'discriminator_epochs': 100,
    'discriminator_alpha': 0.0001,
    'fvp_damping': 0.1,
    'd-steps': 10,
    'vae_epochs': 100,
    'vae_folds': 10,
    'bcloning_epochs': 100,
    'bcloning_folds': 5,
    'env_reward_lambda': 0.01,
    'plot_dir': './plots'
}
