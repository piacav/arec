from sys import platform
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC

# Inizialization variables
sbagliati = 0
indovinati = 0
unrecognized = 0
rigacm = None
colonnacm = None
ts, tr, data, labelList, train_images, test_images = [], [], [], [], [], []
age_class = [14, 24, 59, 100]
classes = {1: "Young",
            2: "Teen",
            3: "Adult",
            4: "Old", }
confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Number of points to be considered as neighbourers
radius = 5
no_points = 2 * radius

# Specify the Haar classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# Creation dataset path for different O.S.
if platform == 'win32':
    dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images/'
else:
    dataset_path = ''

# Constuct the figure for histogram
#plt.style.use("ggplot")
#(fig, ax) = plt.subplots()
#fig.suptitle("Local Binary Patterns")
#plt.ylabel("% of Pixels")
#plt.xlabel("LBP pixel bucket")

# Function for resize image
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

# Create an image cropped on face detected by haar classifier
def rect_create (faces_rect):
    global face_img
    for (x, y, w, h) in faces_rect:
        # Create rectangle on image
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
        a = y + 1
        b = (y + h)-1
        c = x + 1
        d = (x + w)-1
        face_img = image_copy[a:b, c:d]

    return face_img

# Function to calculate the class of age
def classifier_age(age):
    if age <= age_class[0]:
        return 1
    elif age <= age_class[1]:
        return 2
    elif age <= age_class[2]:
        return 3
    else:
        return 4

# Create test set and train set importing document txt
train_file = open("train_set.txt", "r")
for i in train_file:
    train_images.append(i.rstrip())

test_file = open("test_set.txt", "r")
for i in test_file:
    test_images.append(i.rstrip())

# __________________________________________________ TRAIN _____________________________________________________________

print('START TRAINING')

for e in train_images:

    # Create image path
    imagePath = dataset_path + e + '.jpg'

    # Read the image
    im = cv2.imread(imagePath)

    # Convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Create a copy of the image to prevent any changes to the original one
    image_copy = gray_image.copy()

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, 1.1, 5)

    if len(faces_rect) == 0:
        continue

    # Plot gray image and wait
    #cv2.imshow("Image", face_img)
    #cv2.waitKey(0)

    # Resize cropped image
    im = resizeImage(rect_create(faces_rect))
    (h, w) = im.shape[:2]
    cellSize = h/10

    # Plot gray image and wait
    #cv2.imshow("Image", im)
    #cv2.waitKey(0)

    # Uniform LBP is used
    lbp = local_binary_pattern(im, no_points, radius, method='uniform')
    # Plot lbp
    #cv2.imshow("LBP", lbp.astype("uint8"))
    #cv2.waitKey(0)

    # Plot histogram
    #ax.hist(lbp.ravel(), density=True, bins=20, range=(0, 256))
    #ax.set_xlim([0, 256])
    #ax.set_ylim([0, 0.030])
    #fig.savefig('temp.png', dpi=fig.dpi)
    #plt.show()
    #cv2.destroyAllWindows()

    # Create histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, no_points + 3), range=(0, no_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Create label
    label = classes.get(classifier_age(int(e[4:6])))

    # Create list of label and list of data for classification
    labelList.append(label)
    data.append(hist)

# Create a model SVC for classification
model = LinearSVC(C=100.0, random_state=42, max_iter=100000)

# Do training
x = model.fit(data, labelList)
#print(data)
print(labelList)

# Save model
#joblib.dump(model, "lbp_model.pkl")

# ___________________________________________________ TEST _____________________________________________________________

print('START TESTING')

# loop over the testing images
for i in test_images:

    # Create image path
    imagePath = dataset_path + i + '.jpg'

    # Read the image
    im = cv2.imread(imagePath)

    # Convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Create a copy of the image to prevent any changes to the original one
    image_copy = gray_image.copy()

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, 1.1, 5)

    if len(faces_rect) == 0:
        unrecognized += 1
        continue

    # Resize cropped image
    im = resizeImage(rect_create(faces_rect))
    (h, w) = im.shape[:2]
    cellSize = h / 10

    # Plot gray image and wait
    # cv2.imshow("Image", im)
    # cv2.waitKey(0)

    # LBP algorithm
    lbp = local_binary_pattern(im, no_points, radius, method='uniform')
    # Plot lbp
    # cv2.imshow("LBP", lbp.astype("uint8"))
    # cv2.waitKey(0)

    # Plot histogram
    # ax.hist(lbp.ravel(), density=True, bins=20, range=(0, 256))
    # ax.set_xlim([0, 256])
    # ax.set_ylim([0, 0.030])
    # fig.savefig('temp.png', dpi=fig.dpi)
    # plt.show()
    # cv2.destroyAllWindows()

    # Create histogram
    (histt, _) = np.histogram(lbp.ravel(), bins=np.arange(0, no_points + 3), range=(0, no_points + 2))
    #print(histt)
    # normalize the histogram
    histt = histt.astype("float")
    histt /= (histt.sum() + 1e-7)
    histNew = np.reshape(histt, (1, len(histt)))

    # Predict image
    prediction = model.predict(histNew)

    # Update Confusion Matrix

    real_age = int(i[4:6])
    print("Real Age  : " + str(real_age))

    classdet = prediction[0]
    print("Class     : " + classdet)

    real_class = classes.get(classifier_age(int(real_age)))
    print("Real Class: " + real_class)

    score = model.decision_function(histNew)[0]
    print("Score     : " + str(score))

    print('#######################')
    confronto = (classdet, real_class)
    if real_class == classdet:
        indovinati += 1
        for key, item in classes.items():
            if item == classdet:
                confusion_matrix[key - 1][key - 1] += 1
                break
    else:
        sbagliati += 1
        for wkey, witem in classes.items():
            if witem == real_class:
                rigacm = wkey
            if witem == classdet:
                colonnacm = wkey
        if rigacm is not None and colonnacm is not None:
            confusion_matrix[rigacm - 1][colonnacm - 1] += 1

    # display the image and the prediction
    #cv2.putText(im, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    #cv2.imshow("Image", im)
    #cv2.waitKey(0)


print('INDOVINATI : ' + str(indovinati))
print('SBAGLIATI  : ' + str(sbagliati))
print('SALTATI    : ' + str(unrecognized))
print('TOTALI     : ' + str(indovinati + sbagliati + unrecognized))
print('CONFUSION MATRIX')
print(np.array(confusion_matrix))
