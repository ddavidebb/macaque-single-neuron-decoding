"""
Script to define the optimal neural network designs resulting from 'Motor decoding from posterior parietal cortex using deep neural networks'.
by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso (submitted to Neurocomputing, 2022).

Author
------
Davide Borra, 2022
"""

from models import FCNN, CNN, RNN
import pickle5 as pickle
import yaml

# target task
target_task = 'reaching_m1'  # or 'reaching_m2', 'reach_to_grasping_light_m3', 'reach_to_grasping_light_m4', 'reach_to_grasping_dark_m3', 'reach_to_grasping_dark_m4'
# target network to initialize
target_network = 'rnn'  # or 'cnn', 'rnn'

# loading optimal hyper-parameters
with open('optimal_hparams.pkl', 'rb') as f:
    optimal_hparams = pickle.load(f)
# loading useful dataset information
with open('info.yml') as file:
    info = yaml.load(file, Loader=yaml.FullLoader)
# model initialization
if target_network == 'fcnn':
    model = FCNN(
        in_chans=info[target_task]['n_cells'],
        n_classes=info[target_task]['n_classes'],
        input_time_length=info[target_task]['input_time_length'],
        n_layers=optimal_hparams['across_motor_tasks']['FCNN']['n_layers'],
        n_units_per_layer=optimal_hparams['across_motor_tasks']['FCNN']['n_units_per_layer'],
        use_bn=optimal_hparams['across_motor_tasks']['FCNN']['use_bn'],
        drop_prob=optimal_hparams['across_motor_tasks']['FCNN']['p_drop']
    )
elif target_network == 'cnn':
    model = CNN(
        in_chans=info[target_task]['n_cells'],
        n_classes=info[target_task]['n_classes'],
        input_time_length=info[target_task]['input_time_length'],
        n_blocks=optimal_hparams['across_motor_tasks']['CNN']['n_blocks'],
        n_conv_per_block=optimal_hparams['across_motor_tasks']['CNN']['n_conv_per_block'],
        n_filter_conv=optimal_hparams['across_motor_tasks']['CNN']['n_filter_conv'],
        filter_size_conv=optimal_hparams['across_motor_tasks']['CNN']['temporal_conv_ks'],
        use_bn=optimal_hparams['across_motor_tasks']['CNN']['use_bn'],
        drop_prob=optimal_hparams['across_motor_tasks']['CNN']['p_drop'],
    )
elif target_network == 'rnn':
    model = RNN(
        in_chans=info[target_task]['n_cells'],
        n_classes=info[target_task]['n_classes'],
        n_layers=optimal_hparams['across_motor_tasks']['RNN']['n_layers'],
        n_hidden_feat_per_layer=optimal_hparams['across_motor_tasks']['RNN']['n_hidden_feat_per_layer'],
        drop_prob=optimal_hparams['across_motor_tasks']['RNN']['p_drop']
    )