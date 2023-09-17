from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator
from utils import append_augmented_data, imshow

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse


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

    # fit image size to M, N sliceable data
    nrow_limit = 28
    ncol_limit = 28
    while nrow_limit % M != 0:
        nrow_limit -= 1
    while ncol_limit % N != 0:
        ncol_limit -= 1

    # Initialize hyperparameters
    lr = 0.0002  # Learning rate
    batch_size = 32
    # z_dim = 100  # Latent vector dimension
    z_dim = nrow_limit * ncol_limit
    n_critic = 1  # Number of critic updates per generator update
    lambda_gp = 10.0  # Gradient penalty coefficient

    # Initialize generator and discriminator
    generator = Generator(z_dim)
    discriminator = Discriminator(z_dim)

    # Define optimizers for generator and discriminator
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

    # Define the loss function (Wasserstein loss)
    def wasserstein_loss(y_real, y_fake):
        return torch.mean(y_real) - torch.mean(y_fake)
    
    
    def gradient_penalty(discriminator, real_data, fake_data):
        # print(real_data.shape, fake_data.shape)
        alpha = torch.rand(real_data.size(0), 1, 1, 1)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.detach().requires_grad = True
        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(d_interpolates.size()),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training loop
    num_epochs = 2000

    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(data_loader):
            # Train discriminator
            discriminator.zero_grad()

            # fit image size to M, N sliceable data
            while real_data.shape[2] % M != 0:
                real_data = real_data[:, :, :-1, :]
            while real_data.shape[3] % N != 0:
                real_data = real_data[:, :, :, :-1]
            
            # Flatten real data
            real_data_flatten = real_data.reshape(batch_size, z_dim)
            
            # Instead of random z_dim vectors, augment real_data
            # z = torch.randn(batch_size, z_dim)
            z = append_augmented_data(real_data, M, N)[batch_size:, :, :, :]
            z_flatten = z.reshape(batch_size, z_dim)
            
            # Generate fake data
            fake_data = generator(z_flatten)
            # fake_data = generator(z)
            
            # Discriminator predictions
            d_real = discriminator(real_data_flatten)
            # d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())
            
            # Calculate Wasserstein loss
            loss_D = wasserstein_loss(d_real, d_fake)

            # Calculate gradient penalty
            # gp = gradient_penalty(discriminator, real_data_flat, fake_data)
            gp = gradient_penalty(discriminator, real_data_flatten, fake_data)

            # Update discriminator loss
            loss_D += lambda_gp * gp
            
            # Update discriminator weights
            loss_D.backward()
            optimizer_D.step()
            
            # # Clip weights (important for WGAN)
            # for p in discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            
            # Train generator every n_critic iterations
            if batch_idx % n_critic == 0:
                generator.zero_grad()
                
                # Instead of random z_dim vectors, augment real_data
                # z = torch.randn(batch_size, z_dim)
                z = append_augmented_data(real_data, M, N)[batch_size:, :, :, :]
                z_flatten = z.reshape(batch_size, z_dim)
                
                # Generate fake data
                fake_data = generator(z_flatten)
                # fake_data = generator(z)
                
                # Discriminator predictions
                d_fake = discriminator(fake_data)
                
                # Generator loss
                loss_G = -torch.mean(d_fake)
                
                # Update generator weights
                loss_G.backward()
                optimizer_G.step()
        print(f"Epoch {epoch + 1}, Discriminator Loss: {loss_D}, Generator Loss: {loss_G}")
    print("Training finished!")
    imshow(torchvision.utils.make_grid(fake_data.reshape(batch_size, 1, nrow_limit, ncol_limit)[:5]))

    # Save the trained model
    torch.save(discriminator.state_dict(), 'model/' + model_name + '_d.pth')
    torch.save(generator.state_dict(), 'model/' + model_name + '_g.pth')
