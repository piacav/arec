
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
from skimage.feature import local_binary_pattern

'''
# List for storing the LBP Histograms, address of images and the corresponding label
X_test = []
X_name = []
#y_test = []
'''

# Constuct the figure for histogram
plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")

def resizeImage(image):
    (h, w) = image.shape[:2]
    width = 360  #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized

def split_dataset(dataset, train_size=0.70):
    train_set = []
    test_set = []
    for v in dataset.values():
        np.random.shuffle(v)
        train_set.append(v[:round(len(v)*train_size)])
        test_set.append(v[round(len(v)*train_size):])
    return train_set, test_set

if platform == 'win32':
    dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images/'
else:
    dataset_path = ''

dataset_dict = {}

for n in range(1, 83):
    dataset_dict[n] = []
for file in os.listdir(dataset_path):
    if not file.startswith('.'):
        persona = int(file[:3])
        dataset_dict[persona].append(file[:-4])

ts, tr = [], []
tr, ts = split_dataset(dataset_dict)
train_images = []
for t in tr:
    for e in t:
        train_images += ([dataset_path + e + '.jpg'])

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Resize the image
    im_res = resizeImage(im)
    (h, w) = im.shape[:2]
    cellSize = h / 10

    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im_res, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", im_gray)
    cv2.waitKey(0)

    # Number of points to be considered as neighbourers
    radius = 5
    no_points = 2 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius)
    cv2.imshow("LBP", lbp.astype("uint8"))
    cv2.waitKey(0)

    # Create histogram
    ax.hist(lbp.ravel(), density=True, bins=20, range=(0, 256))
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 0.030])
    fig.savefig('temp.png', dpi=fig.dpi)
    plt.show()
    cv2.destroyAllWindows()

    # Calculate the histogram
    x = np.unique(lbp.ravel())

    # Normalize the histogram
    hist = x / sum(x)
    hist = hist.reshape(1, -1)

    def extractHistogramList(image, label):

        # get the descriptor class initiated
        histogramList = []
        LabelList = []

        # This mask has the same width and height a the original image and has a default value of 0 (black).
        maskedImage = np.zeros(image.shape[:2], dtype="uint8")
        ########### create imageROIList here ############

        (h, w) = image.shape[:2]

        # Divide the image into 100 pieces
        cellSizeYdir = h / 10
        cellSizeXdir = w / 10

        # start in origo
        x = 0
        y = 0
        counterInt = 0

        # 10*10 = 100
        for i in range(10):

            # update this value
            y = cellSizeYdir * (i)

            x = 0  # it starts at 0 for a new row
            for j in range(10):
                print ("[x] inspecting imageROI %d" % (counterInt))
                counterInt = counterInt + 1

                x = cellSizeXdir * (j)

                imageROI = image[y: cellSizeYdir * (i + 1), x:cellSizeXdir * (j + 1)]

                # print "ystart  " + str(y) + "  yjump  " + str((cellSizeYdir * (i+1)))
                # print "xstart  " + str(x) +  "  xjump  " + str((cellSizeXdir * (j+1)))

                # grayscale and calculate histogram
                grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
                hist = lbp.describe(grayImageROI)

                histogramList.append(hist)
                LabelList.append(label)

        return histogramList, LabelList


    histogramList, LabelList = extractHistogramList(im_gray, 'histo')

    '''
   

nrows = 1
ncols = 1
fig, axes = plt.subplots(nrows,ncols, squeeze=False)
for row in range(nrows):
    for col in range(ncols):
        axes[row][col].imshow(cv2.cvtColor(cv2.imread(X_name[row*ncols+col]), cv2.COLOR_BGR2RGB))
        axes[row][col].axis('off')
        axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))
'''

