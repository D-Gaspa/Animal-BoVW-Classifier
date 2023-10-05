import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class BOVW:
    def __init__(self, data_path, num_clusters=50):
        self.data_path = data_path
        self.num_clusters = num_clusters
        self.descriptor_list = []
        self.labels = np.array([])
        self.label_names = []
        self.label_count = 0
        self.clf = SVC()

    def fit(self):
        sift = cv2.SIFT.create()
        image_labels = []
        image_features = []

        print("Starting feature extraction")

        # Iterate through each class folder
        for class_name in os.listdir(self.data_path):
            self.label_names.append(class_name)
            class_path = os.path.join(self.data_path, class_name)

            print(f"Extracting features from class {class_name}")
            # Iterate through each image in the class folder

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, 0)  # read the image in grayscale

                mask = np.ones_like(img, np.uint8) * 255  # creating a mask of the same size with all features included
                kp, des = sift.detectAndCompute(img, mask)  # get key points and descriptors

                if des is not None:  # check if descriptors are extracted
                    self.descriptor_list.append(des)
                    self.labels = np.append(self.labels, np.full(des.shape[0], self.label_count))
                    image_features.append(des)
                    image_labels.append(self.label_count)

                    print(f"Extracted {des.shape[0]} descriptors from {img_name} in class {class_name}")

            self.label_count += 1

        print("Feature extraction complete")

        # Stack all the descriptors vertically in a numpy array
        descriptors = np.vstack(self.descriptor_list)

        # Perform k-means clustering
        print("Starting K-Means Clustering")
        k_means = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, n_init=3)
        k_means.fit(descriptors)
        print("K-Means Clustering complete")

        # Create histograms of visual words
        print("Creating histograms of visual words")
        image_histograms = [np.bincount(k_means.predict(im_features), minlength=self.num_clusters)
                            for im_features in image_features]
        image_histograms = np.array(image_histograms)
        print("Histograms of visual words created")

        # Split into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(
            image_histograms, image_labels, test_size=0.3, random_state=42)

        # Train the SVM
        print("Training the SVM")
        self.clf.fit(x_train, y_train)
        print("SVM trained")

        # Evaluate
        print("Evaluating the SVM")
        y_pred = self.clf.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=self.label_names)  # Store the report in a variable
        print(report)  # Print the classification report
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix for better visualization
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=self.label_names, yticklabels=self.label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Get the current date and time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format the timestamp

        # Save the confusion matrix plot with the timestamp in the filename
        plt.savefig(f'../../results/classification_results/confusion_matrix_{timestamp}.png')
        plt.show()

        print("Accuracy: ", accuracy_score(y_test, y_pred))

        # Save the classification report as a text file with the timestamp in the filename
        with open(f'../../results/classification_results/classification_report_{timestamp}.txt', 'w') as f:
            f.write(report)


if __name__ == "__main__":
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'
    base_data_dir = os.path.join('..', '..', 'data')
    bovw = BOVW(os.path.join(base_data_dir, 'enhanced_images'))
    bovw.fit()
