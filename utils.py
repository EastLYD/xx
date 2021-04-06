from torch.utils import data
from scipy.io import loadmat
import torch
import os
import numpy as np


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)


    def read_mats(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:

            mat = loadmat(os.path.join(path, selected_folder, 'frame_{:02d}.mat'.format(i)))
            if use_transform is not None:
                mat = use_transform(mat)
            mat=mat['rai_d64']
            mat = torch.from_numpy(mat).float()
            #mat=np.log10(mat)

            mat=mat.permute(2,0,1)
            mat = mat[0,:,:].unsqueeze_(0)
            X.append(mat)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
       # X = self.read_mats(self.data_path, folder, self.transform)     # (input) spatial images
        X = self.read_mats(self.data_path, folder,None)


        print(X.shape)
        return X









# import imageio
# # Receives [3,32,64,64] tensor, and creates a gif
# def make_gif(images, filename):
#     x = images.permute(1,2,3,0)
#     x = x.numpy()
#     frames = []
#     for i in range(32):
#         frames += [x[i]]
#     imageio.mimsave(filename, frames)
#
# #x = torch.rand((3,32,64,64))
# #make_gif(x, 'movie.gif')
