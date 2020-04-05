import os
from sys import platform
import numpy as np

# Inizialization variables
dataset_dict_age, dataset_dict_gen = {}, {}
dataset_list, train_images, test_images = [], [], []

# Create dataset path
dataset_path = 'images'

# Function to do random split between training set and test set
def split_dataset(dataset, train_size=0.80):
    train_set = []
    test_set = []
    for v in dataset.values():
        np.random.shuffle(v)
        train_set.append(v[:round(len(v)*train_size)])
        test_set.append(v[round(len(v)*train_size):])
    return train_set, test_set

# Dataset dict for age nd gender inizialization values
for n in range(1, 4):
    dataset_dict_age[n] = []
for m in range(1, 3):
    dataset_dict_gen[m] = []

# Add value on dict for age with different classifications
for file in os.listdir(dataset_path):
    if not file.startswith('.'):
        eta = int(file[4:6])
        if eta <= 10:
            dataset_dict_age[1].append(file[:-4])
        elif eta <= 30:
            dataset_dict_age[2].append(file[:-4])
        else:
            dataset_dict_age[3].append(file[:-4])

# Add value on dict for gender with M of F classification
for file in os.listdir(dataset_path):
    if not file.startswith('.'):
        gen = file[-1]
        if gen == 'M':
            dataset_dict_gen[1].append(file[:-4])
        else:
            dataset_dict_gen[2].append(file[:-4])

# Split dataset with random function. Change dataset for change split type
tr, ts = split_dataset(dataset_dict_age)

# Create train images list with images name
for t in tr:
    for e in t:
        train_images += ([e])

# Create test images list with images name
for tt in ts:
    for e in tt:
        test_images += ([e])

# Open txt documents for train set and test set
test = open("test_set.txt", "w+")
train = open("train_set.txt", "w+")

# Write on documents the images name
for i in test_images:
    test.write(str(i) + "\n")

for j in train_images:
    train.write(str(j) + "\n")

# Close documents
train.close()
test.close()

# Print elements in documents
print("TRAIN SET:\n" + str(train_images) + "\nTEST SET : \n" + str(test_images))