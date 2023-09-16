from PIL import Image
from pathlib import Path
from utils import *

import numpy as np
import argparse
import random


class ImageCutter:
    def __init__(self, row_slice, col_slice):
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.img_segments = []
        self.img_path = None
        self.img_name = None

    def cut_image(self, img_path, is_augment=False, reps=1):
        self.img_path = img_path
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # shrink img array until fit row_slice, col_slice
        while img_array.shape[0] % self.row_slice != 0:
            img_array = img_array[:-1, :]
        while img_array.shape[1] % self.col_slice != 0:
            img_array = img_array[:, :-1]

        for _ in range(reps):
            segments_ls = []
            # save img segments into instance variable self.img_segments
            for i in range(self.row_slice):
                for j in range(self.col_slice):
                    segment = img_array[
                        i*(img_array.shape[0]//self.row_slice):(i+1)*(img_array.shape[0]//self.row_slice),
                        j*(img_array.shape[1]//self.col_slice):(j+1)*(img_array.shape[1]//self.col_slice)
                        ]

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
            self.img_segments.append(segments_ls)

    def save_segments(self, output_path=None):
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            print(f"Directory '{output_path}' created successfully.")

            # save cut images
            for i, ls in enumerate(self.img_segments):
                r_smpl = random.sample(range(len(ls)), len(ls))
                segment_sample_path = output_path + "/" + "{}".format(i)
                Path(segment_sample_path).mkdir(parents=True, exist_ok=True)
                for ix, img_arr in enumerate(ls):
                    img = Image.fromarray(img_arr)
                    # file_dir_path = segment_sample_path + "/" + "{}".format(r_smpl[ix])
                    # Path(file_dir_path).mkdir(parents=True, exist_ok=True)
                    # file_path =file_dir_path + "/" + "{}".format(r_smpl[ix]) + "/" + "{}".format(r_smpl[ix]) + ".jpg"
                    file_path =segment_sample_path + "/" + "{}".format(r_smpl[ix]) + ".jpg"
                    img.save(file_path)
                    print(f"Image saved as {file_path}")
        else:
            # create directory/folder to save cut images
            ix_last_period = self.img_path[::-1].index(".")
            # img_path에서 확장자명 (.jpg) 뺀 path
            # e.g., cifar_10_jpeg_1000/7_output
            directory_path = self.img_path[:-(ix_last_period+1)]

            Path(directory_path).mkdir(parents=True, exist_ok=True)
            print(f"Directory '{directory_path}' created successfully.")

            # save cut images
            for i, ls in enumerate(self.img_segments):
                r_smpl = random.sample(range(len(ls)), len(ls))
                segment_sample_path = directory_path + "/" + "{}".format(i)
                Path(segment_sample_path).mkdir(parents=True, exist_ok=True)
                print(f"Directory '{segment_sample_path}' created successfully.")
                for ix, img_arr in enumerate(ls):
                    img = Image.fromarray(img_arr)
                    # file_dir_path = segment_sample_path + "/" + "{}".format(r_smpl[ix])
                    # Path(file_dir_path).mkdir(parents=True, exist_ok=True)
                    # print(f"Directory '{file_dir_path}' created successfully.")
                    # file_path =file_dir_path + "/" + "{}".format(r_smpl[ix]) + "/" + "{}".format(r_smpl[ix]) + ".jpg"
                    file_path =segment_sample_path + "/" + "{}".format(r_smpl[ix]) + ".jpg"
                    img.save(file_path)
                    print(f"Image saved as {file_path}")


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    # Define the arguments you want to accept
    parser.add_argument("arg1", type=str, help="Path of image to cut and augment.")
    parser.add_argument("arg2", type=int, help="Into how many slices in row to cut original image.")
    parser.add_argument("arg3", type=int, help="Into how many slices in col to cut original image.")
    parser.add_argument("arg4", type=str, help="Whether to apply augmentation to each image segments.")
    parser.add_argument("arg5", type=int, help="How many sets of cut segments to create.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments using args.arg{x}
    img_path = args.arg1
    M = args.arg2
    N = args.arg3
    is_augment = True if args.arg4 == "True" else False
    reps = args.arg5

    # Create an ImageCutter object
    image_cutter = ImageCutter(M, N)

    # Cut images
    image_cutter.cut_image(img_path, is_augment=is_augment, reps=reps)

    # Save cut images
    image_cutter.save_segments()
