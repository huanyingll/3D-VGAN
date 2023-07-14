'''
params.py

Managers of all hyper-parameters

'''

import torch
mode=1
epochs = 36000
batch_size = 2
soft_label = False
adv_weight = 0
d_thresh = 0.8
z_dim = 200
z_dis = "uni"
model_save_step = 400
g_lr = 1e-5
d_lr = 1e-6
"""
g_lr = 1e-4
d_lr = 1e-5
"""
"""
g_lr = 1e-5
d_lr = 5e-6
"""
beta = (0.5, 0.999)
cube_len = 128
leak_value = 0.2
bias = False
model_dir = 'models/'
output_dir = '../outputs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)
    print('pytoch=',torch.__version__)
    print(l * '*' + 'hyper-parameters' + l * '*')
