from sys import platform
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import classification_report
from os import path

# Inizialization variables
wrong_age, correct_age, wrong_gen, correct_gen, unrecognized = 0, 0, 0, 0, 0
r_age, c_age, r_gen, c_gen = None, None, None, None
ts, tr, data, labelList_age, labelList_gen, train_images, test_images, \
y_test_age, y_pred_age, y_test_gen, y_pred_gen = \
    [], [], [], [], [], [], [], [], [], [], []
age_param = [10, 50, 100]
classes_age = {1: "Children",
               2: "Young",
               3: "Adult"}
classes_gen = {1: "Male",
               2: "Female"}
confusion_matrix_age = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
confusion_matrix_gen = [[0, 0],
                       [0, 0]]

# Number of points to be considered as neighbourers
radius = 5
no_points = 2 * radius

# Specify the Haar classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# Creation dataset path for different O.S.
if platform == 'win32':
    dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images2\\'
    #dataset_path = 'D:\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images2/'
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
    if age <= age_param[0]:
        return 1
    elif age <= age_param[1]:
        return 2
    else:
        return 3

# Return the appropriate string for gender class
def classifier_gender(gender):
    return 'Male' if gender.upper() == 'M' else 'Female'

# Create test set and train set importing document txt
train_file = open("train_set.txt", "r")
for i in train_file:
    train_images.append(i.rstrip())

test_file = open("test_set.txt", "r")
for i in test_file:
    test_images.append(i.rstrip())

# __________________________________________________ TRAIN _____________________________________________________________

if not path.exists("lbp_model_age.pkl") or not path.exists("lbp_model_gen.pkl"):
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
        label_age = classes_age.get(classifier_age(int(e[4:6])))
        label_gen = classifier_gender(str(e[-1]))

        # Create list of label and list of data for classification
        labelList_age.append(label_age)
        labelList_gen.append(label_gen)
        data.append(hist)
    print(labelList_age)
    print(labelList_gen)
    # Create a model SVC for age and gender classification
    model_age = LinearSVC(C=200.0, random_state=42, max_iter=100000)
    model_gen = LinearSVC(C=200.0, random_state=42, max_iter=100000)

    # Do training
    x = model_age.fit(data, labelList_age)
    y = model_gen.fit(data, labelList_gen)
    #print(data)
    print(labelList_age)
    print(labelList_gen)
    # Save model
    dump(model_age, "lbp_model_age.pkl")
    dump(model_gen, "lbp_model_gen.pkl")

else:
    model_age = load("lbp_model_age.pkl")
    model_gen = load("lbp_model_gen.pkl")
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
    #cv2.imshow("Image", im)
    #cv2.waitKey(0)

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
    prediction_age = model_age.predict(histNew)
    prediction_gen = model_gen.predict(histNew)

    # Age data
    real_age = int(i[4:6])
    print("Real Age  : " + str(real_age))

    age_det = prediction_age[0]
    print("Age Class : " + age_det)

    real_age_class = classes_age.get(classifier_age(int(real_age)))
    print("Real age Class: " + real_age_class)

    age_score = model_age.decision_function(histNew)[0]
    #print("Age Score : " + str(age_score))

    # Gender data
    real_gen = classifier_gender(i[-1])
    print('Real Gen   : ' + real_gen)

    gen_det = prediction_gen[0]
    print('Gen Pred : ' + gen_det)

    gen_score = model_gen.decision_function(histNew)[0]
    print("Gen Score : " + str(gen_score))

    print('#######################')

    y_pred_age.append(age_det)
    y_test_age.append(real_age_class)
    y_pred_gen.append(gen_det)
    y_test_gen.append(real_gen)

    # Age check
    if real_age_class == age_det:
        correct_age += 1
        for key, item in classes_age.items():
            if item == age_det:
                confusion_matrix_age[key - 1][key - 1] += 1
                break
    else:
        wrong_age += 1
        for wkey, witem in classes_age.items():
            if witem == real_age_class:
                r_age = wkey
            if witem == age_det:
                c_age = wkey
        if r_age is not None and c_age is not None:
            confusion_matrix_age[r_age - 1][c_age - 1] += 1

    # Gender check
    if real_gen == gen_det:
        correct_gen += 1
        for key, item in classes_gen.items():
            if item == gen_det:
                confusion_matrix_gen[key - 1][key - 1] += 1
                break
    else:
        wrong_gen += 1
        for wkey, witem in classes_gen.items():
            if witem == real_gen:
                r_gen = wkey
            if witem == gen_det:
                c_gen = wkey
        if r_gen is not None and c_gen is not None:
            confusion_matrix_gen[r_gen - 1][c_gen - 1] += 1

    # display the image and the prediction
    #cv2.putText(im, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    #cv2.imshow("Image", im)
    #cv2.waitKey(0)

# Compute metrics for performance evaluation
cmarray_age = np.array(confusion_matrix_age)
cmarray_gen = np.array(confusion_matrix_gen)

TruePositive_age = np.diag(cmarray_age)

FalsePositive_age, FalseNegative_age, TrueNegative_age, Accuracy_age = [], [], [], []
FalsePositive_gen, FalseNegative_gen, TrueNegative_gen, Accuracy_gen = [], [], [], []

for ifp in range(3):
    FalsePositive_age.append(sum(cmarray_age[:, ifp]) - cmarray_age[ifp, ifp])

for ifn in range(3):
    FalseNegative_age.append(sum(cmarray_age[ifn, :]) - cmarray_age[ifn, ifn])

for itn in range(3):
    temp = np.delete(cmarray_age, itn, 0)  # delete ith row
    temp = np.delete(temp, itn, 1)  # delete ith column
    TrueNegative_age.append(sum(sum(temp)))

for c in range(3):
    Accuracy_age.append((TruePositive_age[c] + TrueNegative_age[c]) / (TruePositive_age[c] +
                                                                       TrueNegative_age[c] +
                                                                       FalsePositive_age[c] +
                                                                       FalseNegative_age[c]))

# Plot non-normalized confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cmarray_age,
                                colorbar=True,
                                class_names=classes_age.items())

print(' ')
print('TOTAL     : ' + str(correct_age + wrong_age + unrecognized))
print('NO FACE   : ' + str(unrecognized))  #erano 81
print(' ')

print('AGE DATA')
print(classification_report(y_test_age, y_pred_age))
print('CONFUSION MATRIX')
print(cmarray_age)
print('CORRECT    : ' + str(correct_age))
print('WRONG      : ' + str(wrong_age))
print('TRUE POSITIVES')
print(TruePositive_age)
print('FALSE POSITIVES')
print(FalsePositive_age)
print('FALSE NEGATIVES')
print(FalseNegative_age)
print('TRUE NEGATIVES')
print(TrueNegative_age)
print('ACCURACY')
print(Accuracy_age)
print(' ')

print('GENDER DATA')
print(classification_report(y_test_gen, y_pred_gen))
print('CONFUSION MATRIX')
print(cmarray_gen)
print('CORRECT    : ' + str(correct_gen))
print('WRONG      : ' + str(wrong_gen))
print(' ')


plt.show()