import torchvision

pretrained_net = torchvision.models.resnet18(pretrained=True)
print(list(pretrained_net.children()))
