import os
import cv2
import numpy as np

def Load_Image(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

natural_images = Load_Image('./dataset/natural_training')
manmade_images = Load_Image('./dataset/manmade_training')

image_size = (128, 128)
natural_images = [cv2.resize(img, image_size) for img in natural_images]
manmade_images = [cv2.resize(img, image_size) for img in manmade_images]

natural_labels = [0] * len(natural_images)
manmade_labels = [1] * len(manmade_images)

data = np.concatenate((natural_images, manmade_images), axis=0)
labels = np.concatenate((natural_labels, manmade_labels), axis=0)

#from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

k = 5
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(data.reshape(len(data), -1), labels)

# Prediction
natural_test = Load_Image('./dataset/natural_test')
manmade_test = Load_Image('./dataset/manmade_test')

natural_test = [cv2.resize(img, image_size) for img in natural_test]
manmade_test = [cv2.resize(img, image_size) for img in manmade_test]

natural_test_labels = [0] * len(natural_test)
manmade_test_labels = [1] * len(manmade_test)

data_test = np.concatenate((natural_test, manmade_test), axis=0)
labels_test = np.concatenate((natural_test_labels, manmade_test_labels), axis=0)

y_pred = knn.predict(data_test.reshape(len(data_test), -1))

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(labels_test, y_pred)
report = classification_report(labels_test, y_pred)

print(f"Accuracy: {accuracy: .2f}")
print(f"Report: \n{report}")