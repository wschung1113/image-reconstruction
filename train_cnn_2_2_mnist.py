from PIL import Image
from utils import append_augmented_data
from cnn_model import Net

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # take in necessary arguments
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

    # Define the arguments you want to accept
    parser.add_argument("arg1", type=str, help="Name of model.")
    parser.add_argument("arg2", type=int, help="Into how many slices in row to cut original image.")
    parser.add_argument("arg3", type=int, help="Into how many slices in col to cut original image.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments using args.arg{x}
    model_name = args.arg1
    M = args.arg2
    N = args.arg3    

    # Define transformations for the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    # Initialize the model
    net = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(5):  # Adjust the number of epochs as needed
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data

            # fit image size to M, N sliceable data
            while inputs.shape[2] % M != 0:
                inputs = inputs[:, :, :-1, :]
            while inputs.shape[3] % N != 0:
                inputs = inputs[:, :, :, :-1]

            # stack augmented data to training batch
            inputs = append_augmented_data(inputs, M, N)

            # first 32 are real MNIST images and latter 32 are augmented MNIST images
            labels = torch.tensor([1]*32+[0]*32)

            # shuffle real and augmented data
            dim_to_shuffle = 0
            permuted_indices = torch.randperm(labels.size(dim_to_shuffle))

            shuffled_inputs = inputs.index_select(dim_to_shuffle, permuted_indices)
            shuffled_labels = labels.index_select(dim_to_shuffle, permuted_indices)

            optimizer.zero_grad()
            outputs = net(shuffled_inputs)
            loss = criterion(outputs, shuffled_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    print("Training finished!")

    # Save the trained model
    torch.save(net.state_dict(), 'model/' + model_name + '.pth')
