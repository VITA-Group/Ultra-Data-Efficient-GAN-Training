import torchvision
from PIL import Image
dataset = torchvision.datasets.CIFAR10("./data", download=True, train=True)
count = 0

for image,label in dataset:
    image.save("cifar10/{}.jpg".format(count))
    count += 1

dataset = torchvision.datasets.CIFAR10("./data", download=True, train=False)
for image,label in dataset:
    image.save("cifar10/{}.jpg".format(count))
    count += 1