import shutil
import os
imagesList = open('images.txt').readlines()
assign = open('train_test_split.txt').readlines()
previous = ''
train_count = 1
test_count = 1
for i in range(len(assign)):
    a,b = imagesList[i].strip().split()
    c,d = assign[i].strip().split()
    current = b.split('/')[0].split('.')[1]
    if (current != previous):
        train_count = 1
        test_count = 1
        previous = current
        if not os.path.exists('train/' + current): 
            os.mkdir('train/' + current)
            os.mkdir('valid/' + current)
    if (d == '1'):
        shutil.copyfile('images/' + b, 'train/{}/{}_{}.jpg'.format(current, current, train_count))
        train_count += 1
    else:
        shutil.copyfile('images/' + b, 'valid/{}/{}_{}.jpg'.format(current, current, test_count))
        test_count += 1