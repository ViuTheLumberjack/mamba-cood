import torch
import matplotlib.pyplot as plt
import os



# id = '000154_0_43'

# id = '000100_0_16'
# id_data = '000154_0_43'
# id_data = '000110_0_21'
id_data = '000116_0_24'

folder_save = 'analysis_feature'

fm_perfect_path = f'{folder_save}/data_perfect/sf_{id_data}.pt'
fm_delay_path = f'{folder_save}/data_delay/sf_{id_data}.pt'
fm_noisy_path = f'{folder_save}/data_noisy/sf_{id_data}.pt'
fm_noisydelay_path = f'{folder_save}/data_noisy_delay/sf_{id_data}.pt'


#open
sp_perfect = torch.load(fm_perfect_path)
sp_delay = torch.load(fm_delay_path)
sp_noisy = torch.load(fm_noisy_path)
sp_noisydelay = torch.load(fm_noisydelay_path)


#number_agents, channels, height, width
ag = 2
data_original = sp_perfect[ag].detach().cpu().numpy()
data_delay = sp_delay[ag].detach().cpu().numpy()
data_noisy = sp_noisy[ag].detach().cpu().numpy()
data_noisydelay = sp_noisydelay[ag].detach().cpu().numpy()


data_original = data_original.sum(0) / data_original.shape[0]
data_delay = data_delay.sum(0) / data_delay.shape[0]
data_noisy = data_noisy.sum(0) / data_noisy.shape[0]
data_noisydelay = data_noisydelay.sum(0) / data_noisydelay.shape[0]

folder = f'analysis_feature/{id_data}/{ag}'
if not os.path.exists(f'{folder}/'):
    os.makedirs(f'{folder}/')


plt.imshow(data_original)
plt.savefig(f'{folder}/data_original.png')

plt.imshow(data_delay)
plt.savefig(f'{folder}/data_delay.png')

plt.imshow(data_noisy)
plt.savefig(f'{folder}/data_noisy.png')

plt.imshow(data_noisydelay)
plt.savefig(f'{folder}/data_noisydelay.png')



#savefig first feature
# plt.imshow(data_original[0,:,:])
# plt.savefig('analysis_feature/sp_1.png')
#
# plt.imshow(data_delay[0,:,:])
# plt.savefig('analysis_feature/sp_delay_1.png')