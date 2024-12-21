# EMPOWERING-IOT-CYBER-NNETWORK-ATTACKS-USING-MACHINE-LEARNING

This project aims to detect and classify network attacks using machine learning (ML) techniques. It leverages a structured dataset in the NetFlow V9 format, which provides network flow data commonly used for network monitoring and security analysis. The focus is on identifying and categorizing three types of attacks: *Port Scanning, **Denial of Service (DoS), and **Malware*, along with normal traffic.

## Project goal
   The goal of this project is to build a machine learning model that is able to detect and classify network attacks. The model will be trained using a dataset of network flows, and will be able to predict if a new flow is an attack or not, and if it is, what kind of attack it is.

## Dataset description
  The dataset used for this project is written in the NetFlow V9 format (format by Cisco, documentation available here). The dataset is composed by two files:

### Training Data (train_net.csv): 
  Includes labeled samples with attack categories (Port scanning, DoS, Malware, or None).
### Testing Data (test_net.csv): 
  Contains samples without the Alert label, used for evaluating predictions.

train_set: ~2 million flows, used for training the model
test_set: ~4 million flows, used for testing the model


The dataset features unique identifiers (FLOW_ID), protocol details, timestamps, and other network flow metrics. Labels in the training set help the ML models learn to differentiate between attack types and normal traffic.

### DATASET LINK:

     https://www.kaggle.com/datasets/ashtcoder/network-data-schema-in-the-netflow-v9-format


## Implementation Details:

### Libraries Used: 
  Python-based libraries like Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, and UMAP for data handling, visualization, and modeling.

### Data Preprocessing:
  - Cleaning and encoding categorical features.
  - Normalizing numerical features to ensure consistent scales.
  - Splitting the data into training and validation subsets.

### Training and Validation*:
  - Models are trained on the labeled dataset and validated using cross-validation to ensure reliability.
  - Metrics like accuracy, precision, recall, and F1-score are used to evaluate performance.


## *Visualization and Insights*:
### UMAP and PCA*:
  Used for visualizing high-dimensional data in 2D or 3D plots, showing clusters for different attack types.
### Model Comparisons*:
  Performance metrics and confusion matrices help identify the most effective model for the task.

### Several machine learning models are employed to achieve robust classification:

1. *K-Nearest Neighbors (KNN)*:
   - A simple, instance-based learning algorithm that classifies a sample based on the majority label of its nearest neighbors.
   - Effective for small datasets but may struggle with large-scale data.

2. *Support Vector Machine (SVC) with RBF Kernel*:
   - Finds the optimal hyperplane to separate data classes.
   - The RBF kernel handles non-linear relationships in the data.

3. *Pipeline with PCA and SVC*:
   - Combines dimensionality reduction (PCA) with SVM for a streamlined and efficient workflow.

4. *Bagging Classifier*:
   - Uses an ensemble of SVM models to improve generalization and robustness by reducing variance.

5. *Random Forest Classifier*:
   - A tree-based ensemble model that builds multiple decision trees and averages their predictions.
   - Handles mixed data types and large datasets effectively.

6. *Extra Trees Classifier*:
   - Similar to Random Forest, but with more randomness in tree splits, leading to faster training.

7. *Neural Network (MLPClassifier)*:
   - A multi-layer perceptron that captures complex patterns in the data.
   - Particularly useful for high-dimensional and non-linear datasets.
  
 
## Output and Predictions"
### Training Output: 
  Trained models can accurately classify whether a network flow contains no attack (None), or if it involves Port Scanning, DoS, or Malware.
### Testing Predictions: 
  Predictions on test_net.csv help assess real-world applicability.

