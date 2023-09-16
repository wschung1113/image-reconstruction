from PIL import Image
from os.path  import join
from pathlib import Path
from obsolete.mnist_data_loader import MnistDataloader

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    from PIL import Image

    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = './'
    training_images_filepath = join(input_path, 'archive/train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 'archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 'archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    for ix, x in enumerate(x_train):
        data = np.array(x)
        
        # Create an Image object from the NumPy array
        image = Image.fromarray(data)
        
        # Specify the file path where you want to save the image
        data_dir = "./data/train/1"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        file_path = data_dir + "/" + "{}.jpg".format(ix)
        
        # Save the image as a JPEG file
        image.save(file_path)

        if ix >= 499:
            break 