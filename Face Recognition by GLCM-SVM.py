#library
import cv2
import numpy as np
import sklearn.neighbors as sn
import skimage.feature as skf
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import imutils
%matplotlib inline

#parameter
featureTr =  [];
labelTr = [];
paraQuantize = 64 
paraAngle = [0, 45, 90, 135]
paraDistance = [1, 2, 3]
test = [1, 2, 3, 9, 10, 11, 17, 18, 19, 20, 25, 26, 27,  33, 34, 35, 41, 42, 43, 49, 50, 51, 57, 58, 59]

# Training Image Loader and Feature Extraction
for _ in range(10):
    for _classname in range(1, 16):
        for _id in test:
            path = '/content/drive/MyDrive/Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm';
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img1 = (img / (256/paraQuantize)).astype(int);
            glcm = skf.greycomatrix(img1, distances=paraDistance, angles=paraAngle, levels=paraQuantize, symmetric=True, normed=True)
            featureCon = skf.greycoprops(glcm, 'contrast')[0]
            featureEne = skf.greycoprops(glcm, 'energy')[0]
            featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
            featureCor = skf.greycoprops(glcm, 'correlation')[0]
            featureTmp = np.hstack((featureCon, featureEne, featureHom, featureCor))
            featureTr.append(featureTmp)
            labelTr.append(_classname)
featureTr= np.array(featureTr)

# Testing Image Loader and Feature Extraction
path = '/content/drive/MyDrive/Tr/emoji/i (1)/t (8).pgm';
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = (img1/ (256/paraQuantize)).astype(int);

glcm = skf.greycomatrix(img, distances=paraDistance, angles=paraAngle, levels=paraQuantize, symmetric=True, normed=True)

featureCon = skf.greycoprops(glcm, 'contrast')[0]
featureEne = skf.greycoprops(glcm, 'energy')[0]
featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
featureCor = skf.greycoprops(glcm, 'correlation')[0]
featureTs = [np.hstack((featureCon, featureEne, featureHom, featureCor))]
labelTs= 2
classifier = svm.SVC(kernel='linear', decision_function_shape='ovo')

classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print('Answer is ' + str(out))

path_origin = '/content/drive/MyDrive/Tr/emoji/i (' + str(out[0]) + ')/t (1).pgm';
pic = cv2.imread(path_origin,cv2.COLOR_BGR2GRAY)

read_input = io.imread(path)
read_origin = cv2.imread(path_origin) 

print(img1.shape)
print(pic.shape)
plt.imshow(read_input)
plt.show()
plt.imshow(read_origin)
plt.show()
plt.figure()
plt.imshow(featureTs)
plt.show()
plt.imshow(featureTr)
plt.show()
