from PIL import Image
from pathlib import Path
from utils import *

import numpy as np
import argparse
import random


def make_full_img_ls(img_ls, seg_ls):
    if not seg_ls:
        out.append(np.array(img_ls))
    else:
        for ix, n in enumerate(seg_ls):
            tmp_ls = seg_ls[:ix] + seg_ls[ix+1:]
            make_full_img_ls(img_ls + [n], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(n)], tmp_ls)
            make_full_img_ls(img_ls + [flip_arr(n)], tmp_ls)
            make_full_img_ls(img_ls + [rotate_90_arr(n, k=3)], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(flip_arr(n))], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(rotate_90_arr(n, k=3))], tmp_ls)
            make_full_img_ls(img_ls + [flip_arr(rotate_90_arr(n, k=3))], tmp_ls)
            make_full_img_ls(img_ls + [mirror_arr(flip_arr(rotate_90_arr(n, k=3)))], tmp_ls)


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
    

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    # Define the arguments you want to accept
    parser.add_argument("arg1", type=str, help="Directory path of cut images.")
    parser.add_argument("arg2", type=int, help="Into how many slices in row to cut original image.")
    parser.add_argument("arg3", type=int, help="Into how many slices in col to cut original image.")

    # Parse the command-line arguments
    args = parser.parse_args()
    segments_path = args.arg1
    M = args.arg2
    N = args.arg3

    # import each JPEG image segment file as np.array
    segments_ls = []
    segments_files = [f for f in Path(segments_path).iterdir() if f.is_file]
    for i in range(len(segments_files)):
        seg_path = segments_path + "/" + segments_files[i].name
        seg_img = Image.open(seg_path)

        # apply every combinations 
        seg_arr = np.array(seg_img)
        segments_ls.append(seg_arr)
        

    # try all possible permutations of segments and pick the permutation with less boundary value distance
    # segments_ls = [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
    out = []
    make_full_img_ls([], segments_ls)

    full_img_arr = []
    for o in out:
        o_arr = make_full_img_arr(o, M, N)
        full_img_arr.append(o_arr)
    
    # TODO
    # infer logit of each full_img_arr to be natural
    # pick full_img_arr arg_max logit