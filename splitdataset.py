import os
from sys import platform
import numpy as np


# Inizialization variables
dataset_dict = {}
dataset_list, train_images, test_images = [], [], []

# Function to do random split between training set and test set
def split_dataset(dataset, train_size=0.70):
    train_set = []
    test_set = []
    for v in dataset.values():
        np.random.shuffle(v)
        train_set.append(v[:round(len(v)*train_size)])
        test_set.append(v[round(len(v)*train_size):])
    return train_set, test_set


# Create dataset path and list
if platform == 'win32':
    #dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
    dataset_path = 'D:\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images/'
else:
    dataset_path = ''

# Dataset dict inizialization values
for n in range(1, 98):
    dataset_dict[n] = []

for file in os.listdir(dataset_path):
    if not file.startswith('.'):
        persona = int(file[:3])
        dataset_dict[persona].append(file[:-4])

tr, ts = split_dataset(dataset_dict)

for t in tr:
    for e in t:
        train_images += ([e])

for tt in ts:
    for e in tt:
        test_images += ([e])

print(train_images)
print(test_images)

test = open("test_set.txt", "w+")
train = open("train_set.txt", "w+")

for i in test_images:
     test.write(str(i) + "\n")

for j in train_images:
     train.write(str(j) + "\n")

train.close()
test.close()



