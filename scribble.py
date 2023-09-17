 # TODO
inputs, labels = next(iter(trainloader))

def imshow(img):
    img = img / 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(inputs[-5:]))

# Create some tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# Append tensor2 to tensor1 along dimension 0 (row-wise)
result_tensor = torch.stack((tensor1, tensor2), dim=0)

print(result_tensor)

import torch

# Create a sample tensor (2D for illustration)
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Specify the dimension along which you want to shuffle (0 for rows, 1 for columns)
dim_to_shuffle = 0

# Generate a random permutation of indices for the specified dimension
permuted_indices = torch.randperm(tensor.size(dim_to_shuffle))

# Use the permutation to shuffle the tensor along the specified dimension
shuffled_tensor = tensor.index_select(dim_to_shuffle, permuted_indices)

print(shuffled_tensor)


import torch

# Create two sample tensors
# tensor1 = torch.randn(32, 1, 28, 28)
tensor1 = torch.randn(32, 1, 28, 28)
tensor2 = torch.randn(1, 1, 28, 28)

# Concatenate them along dimension 0 (vertical stacking)
concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)

print(concatenated_tensor.shape)

torch.cat((concatenated_tensor, tensor2), dim=0).shape



batch_size = 1
model_path = "model/cnn_model.pth"
M = 2
N = 2
real_data, _ = next(iter(data_loader))


is_augment = True
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# fetch pre-trained model
model = Net()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

model.eval()
data = next(iter(testloader))
X, _ = data


trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

X, _ = next(iter(trainloader))

Image.fromarray(np.array(X[0][0])).show()
Image.fromarray(np.array(X[0][0])*0.5+0.5).show()


inputs, labels = next(iter(testloader))

def imshow(img):
    img = img * 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(torch.cat((X, ans), dim=0)))
imshow(torchvision.utils.make_grid(ans))

imshow(torchvision.utils.make_grid(z[:16]))



import torch

# Create a 2D tensor
tensor = torch.tensor([[2, 7, 1],
                       [8, 4, 9]])

# Find the argmax along dimension 1 (column-wise)
argmax_indices = torch.argmax(tensor, dim=0)

print(argmax_indices)  # This will give you a tensor of indices along dimension 1


torch.argmax(outputs, dim=0)



