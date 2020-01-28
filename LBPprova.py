# OpenCV bindings
import cv2
# To performing path manipulations
import os
from sklearn.model_selection import train_test_split as tts
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import csv
import matplotlib.pyplot as plt
import numpy as np

def split_dataset(dataset, train_size=0.70):
    train_set = []
    test_set = []
    for person in dataset:
        np.random.shuffle(person)
        train_set.append(person[:round(len(person)*train_size)])
        test_set.append(person[round(len(person)*train_size):])
    return train_set, test_set

# List for storing the LBP Histograms, address of images and the corresponding label
X_test = []
X_name = []
#y_test = []
dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images'
dataset_dict = {}

for n in range(1, 83):
    dataset_dict[n] = []
for file in os.listdir(dataset_path):
    dataset_dict[int(file[:3])].append(file[:-4])

train_set, test_set = []
for v in dataset_dict.keys():
    tr, ts = split_dataset()




#x_train, y_train, x_test, y_test = tts()

train_images = ['C:\\Users\\andry\\Desktop\\FGNET\\images\\001A19.jpg', 'C:\\Users\\andry\\Desktop\\FGNET\\images\\001A43a.jpg']
#train_images = ['/Users/piacavasinni/Desktop/FGNET/images/001A19.jpg', '/Users/piacavasinni/Desktop/FGNET/images/001A43a.jpg']
#train_images = ['/Users/piacavasinni/Desktop/FotoDB/grigios.png','/Users/piacavasinni/Desktop/FotoDB/lightgry.jpg']
# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    plt.imshow(lbp, cmap='gray')
    plt.show()
    # Calculate the histogram
    x = np.unique(lbp.ravel())
    # Normalize the histogram
    hist = x/sum(x)
    print(lbp.ravel())
    print(hist)

    plt.hist(lbp.ravel(), 256, [0, 256])
    plt.title(train_image)
    plt.show()
    # Append image path in X_name
    X_name.append(train_image)
    # Append histogram to X_name
    X_test.append(hist)

nrows = 1
ncols = 1
fig, axes = plt.subplots(nrows,ncols, squeeze=False)
for row in range(nrows):
    for col in range(ncols):
        axes[row][col].imshow(cv2.cvtColor(cv2.imread(X_name[row*ncols+col]), cv2.COLOR_BGR2RGB))
        axes[row][col].axis('off')
        axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))
plt.show()