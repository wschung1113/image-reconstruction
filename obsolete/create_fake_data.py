from obsolete.cut_image import ImageCutter
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
            new_img_ls = img_ls + [n]
            tmp_ls = seg_ls[:ix] + seg_ls[ix+1:]
            make_full_img_ls(new_img_ls, tmp_ls)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    # Define the arguments you want to accept
    parser.add_argument("arg1", type=str, help="Path of directory with original jpg images.")
    parser.add_argument("arg2", type=int, help="Into how many slices in row to cut original image.")
    parser.add_argument("arg3", type=int, help="Into how many slices in col to cut original image.")
    parser.add_argument("arg4", type=bool, help="Whether to apply augmentation to each image segments.")
    parser.add_argument("arg5", type=int, help="How many sets of cut segments to create.")
    parser.add_argument("arg6", type=int, help="Threshold of how many permutations to use.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments using args.arg{x}
    img_dir_path = args.arg1
    M = args.arg2
    N = args.arg3
    is_augment = args.arg4
    reps = args.arg5
    permu_threshold = args.arg6

    image_cutter = ImageCutter(M, N)

    # process all images in the directory
    orig_images = [f for f in Path(img_dir_path).iterdir() if f.suffix == ".jpg"]
    
    data_dir = "data/train/2"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for f in orig_images:
        image_cutter.cut_image(f, is_augment=is_augment, reps=reps)
        for i, ls in enumerate(image_cutter.img_segments): # image_cutter.img_segments is a list of lists of img segments
            out = []
            make_full_img_ls([], ls)
            for j, o in enumerate(out):
                o_arr = make_full_img_arr(o, M, N)
                img = Image.fromarray(o_arr)
                fake_img_path = data_dir + "/" + "fake_{}_".format(i) + "{}_".format(j) + f.name
                img.save(fake_img_path)
                if j == permu_threshold - 1:
                    break
