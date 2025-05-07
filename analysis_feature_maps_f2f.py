import torch
import matplotlib.pyplot as plt
import os
import numpy as np

#interesting example: 2309, 3100, 4508

folder_base = 'FEATURE_SAVED/feature_saved_f2f'
type_data = 'train'
scenario_index = 7
ag = 9233

folder_data = f'{folder_base}/{type_data}/{scenario_index}/{ag}'
files = os.listdir(folder_data)
files = sorted(files)

for file in files:
    #create folder of exmaple with scenario index and ag
    if not os.path.exists(f'IMAGES_F2F/{scenario_index}/{ag}'):
        os.makedirs(f'IMAGES_F2F/{scenario_index}/{ag}')
    folder_save = f'IMAGES_F2F/{scenario_index}/{ag}'


    data = np.load(f'{folder_data}/{file}')
    data = data['features']
    data = data.sum(0) / data.shape[0]
    data = data.astype(np.float32)
    plt.imshow(data)
    #remove .npz ext
    file = file[:-4]
    plt.savefig(f'{folder_save}/{file}.png')
    print(f'{folder_save}/{file}.png')




















# id_data = '3100'

# data_perfect_path = f'FEATURE_SAVED/feature_saved_compressed/train/features_{id_data}.npz'
# data_delay_path = f'FEATURE_SAVED/feature_saved_compressed_delay200/train/features_{id_data}.npz'

# #open
# data_perfect = np.load(data_perfect_path)
# data_perfect = data_perfect['features']
# data_delay = np.load(data_delay_path)
# data_delay = data_delay['features']


# #number_agents, channels, height, width
# ag = 1
# data_perfect_ag = data_perfect[ag].copy()
# data_delay_ag = data_delay[ag].copy()

# #float 32
# data_perfect_ag = data_perfect_ag.astype(np.float32)
# data_delay_ag = data_delay_ag.astype(np.float32)

# data_perfect_ag = data_perfect_ag.sum(0) / data_perfect_ag.shape[0]
# data_delay_ag = data_delay_ag.sum(0) / data_delay_ag.shape[0]

# #create folder of exmaple
# if not os.path.exists(f'IMAGES/{id_data}'):
#     os.makedirs(f'IMAGES/{id_data}')
# folder_save = f'IMAGES/{id_data}'

# #difference abs
# difference = np.abs(data_perfect_ag - data_delay_ag)
# #set threshold
# threshold = 1.0
# difference[difference < threshold] = 0
# difference[difference >= threshold] = 1
# #plt
# plt.imshow(difference)
# plt.savefig(f'{folder_save}/difference_ag{ag}_threshold{threshold}.png')

# plt.imshow(data_perfect_ag)
# plt.savefig(f'{folder_save}/dataOriginal_ag{ag}.png')

# plt.imshow(data_delay_ag)
# plt.savefig(f'{folder_save}/dataDelay_ag{ag}.png')

# plt.imshow(difference)
# plt.savefig(f'{folder_save}/difference_ag{ag}.png')

# print('print images!')

