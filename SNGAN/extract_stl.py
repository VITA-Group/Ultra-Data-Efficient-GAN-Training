import torchvision
from PIL import Image
dataset = torchvision.datasets.STL10("./data", download=True, split="train")
count = 0

for image,label in dataset:
    image.save("stl/{}.jpg".format(count))
    count += 1
    