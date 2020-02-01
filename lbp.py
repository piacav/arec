import os
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
from skimage.feature import local_binary_pattern
from sklearn.externals import joblib


labelList = []
data = []

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

def split_dataset(dataset, train_size=0.90):
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
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images3/'
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
test_images = []
for t in tr:
    for e in t:
        train_images += ([e])

for tt in ts:
    for e in tt:
        test_images += ([e])

for e in train_images:
    # Read the image
    imagePath = dataset_path + e + '.jpg'
    im = cv2.imread(imagePath)
    im = resizeImage(im)
    (h, w) = im.shape[:2]
    cellSize = h/10
    # Resize the image
    #im_res = resizeImage(im)
    #(h, w) = im.shape[:2]
    #cellSize = h / 10

    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", im_gray)
    #cv2.waitKey(0)

    # Number of points to be considered as neighbourers
    radius = 5
    no_points = 2 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius)
    #cv2.imshow("LBP", lbp.astype("uint8"))
    #cv2.waitKey(0)

    # Create histogram
    #ax.hist(lbp.ravel(), density=True, bins=20, range=(0, 256))
    #ax.set_xlim([0, 256])
    #ax.set_ylim([0, 0.030])
    #fig.savefig('temp.png', dpi=fig.dpi)
    #plt.show()
    #cv2.destroyAllWindows()

    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, no_points + 3),
                             range=(0, no_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    labelList.append(e[:3])
    data.append(hist)

model = LinearSVC(C=100.0, random_state=42, max_iter=100000)
#print(data)
#print(labelList)
x = model.fit(data, labelList)
joblib.dump(model, "lbp_model.pkl")

sbagliati = 0
indovinati = 0
# loop over the testing images
for i in test_images:
    imagePath = dataset_path + i + '.jpg'
    # load the image, convert it to grayscale, describe it,
    # and classify it
    imagetest = cv2.imread(imagePath)
    imagetest = resizeImage(imagetest)
    (h, w) = imagetest.shape[:2]
    cellSize = h / 10
    graytest = cv2.cvtColor(imagetest, cv2.COLOR_BGR2GRAY)

    radius = 5
    no_points = 2 * radius
    # Uniform LBP is used
    lbptest = local_binary_pattern(graytest, no_points, radius)
    (histt, _) = np.histogram(lbptest.ravel(), bins=np.arange(0, no_points + 3), range=(0, radius + 2))

    # Normalize the histogram
    histt = histt.astype("float")
    histt /= (histt.sum() + 1e-7)
    histNew = np.reshape(histt, (1, len(histt)))
    prediction = model.predict(histNew)
    score = model.decision_function(histNew)
    #prediction = model.predict(histt.reshape(1, -1))
    #score = model.decision_function(histt.reshape(1,-1))


    if (int(i[:3])) == (int(prediction[0])):

        indovinati += 1
        cv2.putText(imagetest, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
    else:
        sbagliati += 1


    # display the image and the prediction
    #cv2.putText(imagetest, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
      #          1.0, (0, 0, 255), 3)
    print(score)
    cv2.imshow("Image", imagetest)
    cv2.waitKey(0)


print('INDOVINATI')
print(indovinati)
print('SBAGLIATI')
print(sbagliati)