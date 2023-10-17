import os
import cv2
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.neural_network import MLPClassifier


class BOVW:
    def __init__(self, data_path, num_clusters=1000):
        self.data_path = data_path
        self.num_clusters = num_clusters
        self.descriptor_list = []
        self.labels = np.array([])
        self.label_names = []
        self.clf = SVC()
        self.mlp = MLPClassifier()
        self.knn = KNeighborsClassifier()
        self.parameters = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 1, 0.1, 0.01]}]
        #self.parameters = [{'solver': ['lbfgs', 'sgd', 'adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'random_state': [50], 'learning_rate_init':[0.001, 0.1, 1.5, 0.00015, 2.1]}]
        #self.parameters = [{"n_neighbors":[5, 9, 8, 7, 3, 4, 10, 20, 30], "weights":['uniform', 'distance'], "n_jobs":[-1]}]
        
        
 
    def fit(self):
        orb = cv2.SIFT.create()
        image_labels = []
        image_features = []

        print("Starting feature extraction")

        # Iterate through each class folder
        for class_idx, class_name in enumerate(os.listdir(self.data_path)):
            self.label_names.append(class_name)
            class_path = os.path.join(self.data_path, class_name)

            print(f"Extracting features from class {class_name}")
            # Iterate through each image in the class folder

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # The image was transform so it can represent from bgr to rgb and then represent that rgb in grayscale

                mask = np.ones_like(img, np.uint8) * 255  # creating a mask of the same size with all features included
                kp, des = orb.detectAndCompute(img, mask)  # get key points and descriptors

                if des is not None:  # check if descriptors are extracted
                    self.descriptor_list.append(des)
                    self.labels = np.append(self.labels, np.full(des.shape[0], class_idx))
                    image_features.append(des)
                    image_labels.append(class_idx)

                    print(f"Extracted {des.shape[0]} descriptors from {img_name} in class {class_name}")


        print("Feature extraction complete")

        # Stack all the descriptors vertically in a numpy array
        descriptors = np.vstack(self.descriptor_list)

        # Perform k-means clustering
        print("Starting K-Means Clustering")
        k_means = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, n_init=3).fit(descriptors)
        print("K-Means Clustering complete")

        # Create histograms of visual words
        print("Creating histograms of visual words")
        image_histograms = [np.bincount(k_means.predict(im_features), minlength=self.num_clusters)
                            for im_features in image_features]
        image_histograms = np.array(image_histograms)
        print("Histograms of visual words created")

        # Split into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(
            image_histograms, image_labels, test_size=0.3, random_state=42, shuffle= True)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        scaler = Normalizer()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        
        # Hyperparameter tuning for SVM
        print("Performing Grid Search for Hyperparameter Tuning")
        grid_search = GridSearchCV(self.clf, self.parameters, cv= 10, n_jobs= -1)
        print("Fitting the SVM")
        grid_search.fit(x_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        best_trainRes = grid_search.best_estimator_

        # Evaluate
        print("Evaluating the SVM")
        y_pred = best_trainRes.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=self.label_names)
        print(report)
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
        classResPath = f'{Path.cwd()}\\results\\classification_results'
        if not os.path.exists(classResPath):
            os.makedirs(classResPath)
        plt.savefig(f'{classResPath}\\confusion_matrix_{timestamp}.png')
        plt.show()

        print("Accuracy: ", accuracy_score(y_test, y_pred))

        # Save the classification report as a text file with the timestamp in the filename
        with open(f'{classResPath}\\classification_report_{timestamp}.txt', 'w') as f:
            f.write(report)
            f.write("\nBest Parameters: ")
            f.write(str(grid_search.best_params_))

def main():
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'
    base_data_dir = f'{Path.cwd()}\\data'
    filtered_images_folder = os.path.join(base_data_dir, 'filtered_images')
    bovw = BOVW(os.path.join(base_data_dir, filtered_images_folder))
    bovw.fit()
    
if __name__ == "__main__":
   main()
