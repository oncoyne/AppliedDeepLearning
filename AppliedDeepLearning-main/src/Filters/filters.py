import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

model = torch.load("epoch_checkpoint_999", map_location=torch.device('cpu'))['model']
filters = model['conv1.weight'].numpy()

plt.rcParams.update({
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

fig, ax = plt.subplots(nrows=6, ncols=6)
for i in range(6):
    for j in range(6):
        f = filters[i + (j * 5), :, :, :]
        f = (f - np.min(f)) / (np.max(f) - np.min(f)) 
        f = np.array(f).transpose((1, 2, 0))
        ax[i][j].imshow(f)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
fig.delaxes(ax[5][2])
fig.delaxes(ax[5][3])
fig.delaxes(ax[5][4])
fig.delaxes(ax[5][5])
   

# show the figure
plt.show()
