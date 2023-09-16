from PIL import Image
from pathlib import Path

import numpy as np
import argparse
import random


def make_full_img_ls(img_ls, seg_ls):
    if not seg_ls:
        out.append(np.array(img_ls))
    else:
        for ix, n in enumerate(seg_ls):
            new_img_ls = img_ls + [n]
            tmp_ls = seg_ls[:ix] + seg_ls[ix+1:]
            make_full_img_ls(new_img_ls, tmp_ls)


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


def get_boundary_distance(full_img_arr, M, N):
    # img_nrow = full_img_arr.shape[0]
    # img_ncol = full_img_arr.shape[1]
    # seg_nrow = img_nrow // M
    # seg_ncol = img_ncol // N
    # distance = 0
    # for i in range(seg_nrow-1, img_nrow-1, seg_nrow):
    #     distance += np.sum(np.square(full_img_arr[i-1:i+1, :] - full_img_arr[i+1:i+3, :]))
    # for j in range(seg_ncol-1, img_ncol-1, seg_ncol):
    #     distance += np.sum(np.square(full_img_arr[:, j-1:j+1] - full_img_arr[:, j+1:j+3]))
    # return distance
    
    distance = 0
    distance += np.sum(np.square(full_img_arr[13, :] - full_img_arr[14, :]))
    distance += np.sum(np.square(full_img_arr[:, 13] - full_img_arr[:, 14]))
    return distance
    

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    # Define the arguments you want to accept
    parser.add_argument("arg1", type=str, help="The first argument")
    parser.add_argument("arg2", type=int, help="The second argument")
    parser.add_argument("arg3", type=int, help="The third argument")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments using args.arg1 and args.arg2
    print(f"Path of directory with segment images: {args.arg1}")
    print(f"Number of row-slices: {args.arg2}")
    print(f"Number of col-slices: {args.arg3}")

    # import each JPEG file as np.array
    segments_path = args.arg1
    M = args.arg2
    N = args.arg3
    segments_ls = []
    segments_files = [f for f in Path(segments_path).iterdir() if f.is_file]
    for i in range(len(segments_files)):
        seg_path = segments_path + "/" + segments_files[i].name
        seg_tmp = Image.open(seg_path)
        segments_ls.append(np.array(seg_tmp))
    # segments_arr = np.array(segments_ls)

    # try all possible permutations of segments and pick the permutation with less boundary value distance
    # tmp = [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
    out = []
    make_full_img_ls([], segments_ls)
    # print(out, out[0], out[0].shape)

    tup_ls = []
    scores = []
    for o in out:
        o_arr = make_full_img_arr(o, M, N)
        o_arr_std = np.std(o_arr)
        o_arr_mean = np.mean(o_arr)
        o_arr_max = np.max(o_arr)
        # o_arr = (o_arr-o_arr_mean) / o_arr_std
        # o_arr = o_arr / o_arr_max
        score = get_boundary_distance(o_arr, M, N)
        tup_ls.append((score, o_arr))
        scores.append(score)
    tup_ls.sort(key=lambda tup: tup[0], reverse=True)
    scores.sort(key=lambda x: x, reverse=True)

    img_arr = tup_ls[0][1]
    # img_arr = tup_ls[0][1] * o_arr_std + o_arr_mean
    # img_arr = tup_ls[0][1] * o_arr_max

    print(scores)
    img = Image.fromarray(img_arr)
    img.show()


img = Image.open("cifar_10_jpeg/7_output.jpg")
img_array = np.array(img)
img_array[10:13, :]
img_array[14:17, :]
img_array[:, 10:13]
img_array[:, 14:17]