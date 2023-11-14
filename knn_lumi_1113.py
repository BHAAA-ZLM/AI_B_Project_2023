import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def compute_rgb_histograms(images, num_bins=27):
    histograms = []
    for img in images:
        hist_red = cv2.calcHist([img], [0], None, [num_bins], [0, 256])
        hist_green = cv2.calcHist([img], [1], None, [num_bins], [0, 256])
        hist_blue = cv2.calcHist([img], [2], None, [num_bins], [0, 256])

        hist_red = hist_red.flatten() / np.sum(hist_red)
        hist_green = hist_green.flatten() / np.sum(hist_green)
        hist_blue = hist_blue.flatten() / np.sum(hist_blue)

        hist = np.concatenate((hist_red, hist_green, hist_blue))
        
        histograms.append(hist)
    return np.array(histograms)

def compute_hog_features(images):
    hog_features = []
    for img in images:
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute HOG features
        features, _ = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

        hog_features.append(features)
    return np.array(hog_features)

def Load_Image(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

natural_images = Load_Image('./dataset/natural_training')
manmade_images = Load_Image('./dataset/manmade_training')

image_size = (256, 256)
natural_images = [cv2.resize(img, image_size) for img in natural_images]
manmade_images = [cv2.resize(img, image_size) for img in manmade_images]

num_histogram_bins = 27 

natural_histograms = compute_rgb_histograms(natural_images, num_bins=num_histogram_bins)
manmade_histograms = compute_rgb_histograms(manmade_images, num_bins=num_histogram_bins)

natural_hog_features = compute_hog_features(natural_images)
manmade_hog_features = compute_hog_features(manmade_images)

data_hist = np.concatenate((natural_histograms, manmade_histograms), axis=0)
data_hog = np.concatenate((natural_hog_features, manmade_hog_features), axis=0)

weight_hist = 0.5
weight_hog = 1 - weight_hist

data_combined = np.concatenate((weight_hist * data_hist, weight_hog * data_hog), axis=1)

labels = np.concatenate(([0] * len(natural_histograms), [1] * len(manmade_histograms)), axis=0)

k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(data_combined, labels)

natural_test = Load_Image('./dataset/natural_test')
manmade_test = Load_Image('./dataset/manmade_test')

natural_test = [cv2.resize(img, image_size) for img in natural_test]
manmade_test = [cv2.resize(img, image_size) for img in manmade_test]

natural_test_histograms = compute_rgb_histograms(natural_test, num_bins=num_histogram_bins)
manmade_test_histograms = compute_rgb_histograms(manmade_test, num_bins=num_histogram_bins)

natural_test_hog_features = compute_hog_features(natural_test)
manmade_test_hog_features = compute_hog_features(manmade_test)

data_test_hist = np.concatenate((natural_test_histograms, manmade_test_histograms), axis=0)
data_test_hog = np.concatenate((natural_test_hog_features, manmade_test_hog_features), axis=0)

data_test_combined = np.concatenate((weight_hist * data_test_hist, weight_hog * data_test_hog), axis=1)

labels_test = np.concatenate(([0] * len(natural_test_histograms), [1] * len(manmade_test_histograms)), axis=0)

y_pred = knn.predict(data_test_combined)

accuracy = accuracy_score(labels_test, y_pred)
report = classification_report(labels_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Report:\n{report}")
