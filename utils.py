import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def make_full_img_arr(img_ls, M, N):
    # e.g., img_ls = [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
    seg_nrow = img_ls[0].shape[0]
    seg_ncol = img_ls[0].shape[1]

    tmp_arr = []
    for i in range(0, len(img_ls), N):
        col_arr = img_ls[i]
        for j in range(1, M):
            col_arr = np.concatenate((col_arr, img_ls[i+j]), axis=1)
        tmp_arr.append(col_arr)

    tmp_arr = np.array(tmp_arr)
    tmp_arr = tmp_arr.reshape(M*seg_nrow, N*seg_ncol)
    return tmp_arr


def mirror_arr(np_arr):
    return np.flip(np_arr, axis=1)


def flip_arr(np_arr):
    return np.flip(np_arr, axis=0)


def rotate_90_arr(np_arr, k):
    return np.rot90(np_arr, k=k)


def make_segments_tensor(data_point, M, N, is_augment=True):
    nrow = data_point.shape[0]
    ncol = data_point.shape[1]
    segments_ls = []
    for i in range(M):
        for j in range(N):
            segment = np.array(data_point[
                i*(nrow//M):(i+1)*(nrow//M),
                j*(ncol//N):(j+1)*(ncol//N)
                ])
            if is_augment:
                u = random.uniform(0, 1)
                if u < 0.5:
                    segment = mirror_arr(segment)
                
                u = random.uniform(0, 1)
                if u < 0.5:
                    segment = flip_arr(segment)
                
                u = random.uniform(0, 1)
                if u < 0.5:
                    segment = rotate_90_arr(segment, k=1)

            segments_ls.append(segment)

    segments_tensor = torch.Tensor(np.array(segments_ls))
    return segments_tensor


def augment_image(data_point, M, N):
    segments_tensor = make_segments_tensor(data_point, M, N)

    # Specify the dimension along which you want to shuffle (0 for rows, 1 for columns)
    dim_to_shuffle = 0

    # Generate a random permutation of indices for the specified dimension
    permuted_indices = torch.randperm(segments_tensor.size(dim_to_shuffle))

    # Use the permutation to shuffle the tensor along the specified dimension
    shuffled_tensor = segments_tensor.index_select(dim_to_shuffle, permuted_indices)

    augmented_data_point = torch.tensor(make_full_img_arr(shuffled_tensor, M, N)).unsqueeze(dim=0).unsqueeze(dim=0)
    return augmented_data_point


def append_augmented_data(batch_data, M, N):
    for i in range(len(batch_data)):
        data_point = batch_data[i][0]
        augmented_data_point = augment_image(data_point, M, N)
        batch_data = torch.cat((batch_data, augmented_data_point), dim=0)
    return batch_data


def imshow(img):
    # img = img / 0.5 + 0.5     # unnormalize
    # img = img * 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
