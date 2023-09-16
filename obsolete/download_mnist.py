import torch
import torchvision
import torchvision.transforms as transforms


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5), (0.5, 0.5))])
# batch_size = 4
# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
# classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


# MNIST 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

batch_size = 1

# 데이터 로더 생성
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)