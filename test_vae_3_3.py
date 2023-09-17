from PIL import Image
from vae_model import VAE
from utils import mirror_arr, flip_arr, rotate_90_arr, make_segments_tensor, augment_image, make_full_img_arr, imshow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse
import numpy as np
import matplotlib.pyplot as plt


def make_full_img_ls(img_ls, seg_ls):
    if not seg_ls:
        out.append(np.array(img_ls))
    else:
        for ix, n in enumerate(seg_ls):
            n = np.array(n)
            tmp_ls = seg_ls[:ix] + seg_ls[ix+1:]
            make_full_img_ls(img_ls + [n], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(n)], tmp_ls)
            make_full_img_ls(img_ls + [flip_arr(n)], tmp_ls)
            make_full_img_ls(img_ls + [rotate_90_arr(n, k=3)], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(flip_arr(n))], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(rotate_90_arr(n, k=3))], tmp_ls)
            make_full_img_ls(img_ls + [flip_arr(rotate_90_arr(n, k=3))], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(flip_arr(rotate_90_arr(n, k=3)))], tmp_ls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    parser.add_argument("arg1", type=str, help="Model weight path to use.")
    parser.add_argument("arg2", type=int, help="Into how many slices in row to cut original image.")
    parser.add_argument("arg3", type=int, help="Into how many slices in col to cut original image.")
    parser.add_argument("arg4", type=str, help="Whether to augment test data or not.")

    args = parser.parse_args()
    
    # model_path = "model/vae_1.pth"
    # M = 3
    # N = 3
    # is_augment = True
    model_path = args.arg1
    M = args.arg2
    N = args.arg3
    is_augment = True if args.arg4 == "True" else False

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # fetch pre-trained model
    model = VAE(x_dim=729, h_dim1= 512, h_dim2=256, z_dim=2)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
                # X, label = next(iter(testloader))
                X, label = data

                # fit image size to M, N sliceable data
                while X.shape[2] % M != 0:
                    X = X[:, :, :-1, :]
                while X.shape[3] % N != 0:
                    X = X[:, :, :, :-1]

                if is_augment:
                    X = augment_image(X[0][0], M, N)

                recon_batch, mu, var = model(X)
                recon_batch = recon_batch.view(1, 1, 27, 27)

                X = X * 0.5 + 0.5
                real_fake = torch.cat((X, recon_batch), dim=0)
                print("Original image is a {}".format(label))
                imshow(torchvision.utils.make_grid(real_fake))

                if i == 0:
                    break