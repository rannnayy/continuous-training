
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans

def dd_cluster_kmeans(initial_data, current_data, use_weights = True):
    model_name = "kmeans"
    print("Start clustering with "+ model_name +" model...")
    
    x_train, x_test = np.array(initial_data).reshape(-1, 1), np.array(current_data).reshape(-1, 1)
    
    clt = KMeans(n_clusters=2, random_state=42, n_init="auto")
    clt.fit(x_train)
    
    cluster_prediction = clt.predict(x_train)
    
    # Get cluster size
    cluster_size = [cluster_prediction.tolist().count(0), cluster_prediction.tolist().count(1)]
    
    # Calculate anomaly scores
    x_train_transformed = clt.transform(x_train)
    if use_weights:
        train_scores = [min(a, b)*cluster_size[[a, b].index(min(a, b))] for a, b in x_train_transformed]
    else:
        train_scores = [min(a, b) for a, b in x_train_transformed]
    
    x_test_transformed = clt.transform(x_test)
    if use_weights:
        test_scores = [min(a, b)*cluster_size[[a, b].index(min(a, b))] for a, b in x_test_transformed]
    else:
        test_scores = [min(a, b) for a, b in x_test_transformed]
    
    return train_scores, test_scores

def dd_classification_svm(initial_data, current_data):
    model_name = "svm"
    print("Start classification with "+ model_name +" model...")
    
    x_train, x_test = np.array(initial_data).reshape(-1, 1), np.array(current_data).reshape(-1, 1)
    
    clf = OneClassSVM()
    clf.fit(x_train)
    
    train_scores = clf.score_samples(x_train)
    test_scores = clf.score_samples(x_test)
    
    return train_scores, test_scores


# Approach 2: Clustering Model
def ttest(ls1, ls2):
    return ttest_ind(ls1, ls2).pvalue < 0.05 # drift true if < 0.05

CLUSTER_PREDICTOR = {
    'kmeans': dd_cluster_kmeans,
    'svm': dd_classification_svm
}

def dd_cluster(model_name, initial_data, current_data):
    initial_scores, current_scores = CLUSTER_PREDICTOR[model_name](initial_data, current_data)
    return ttest(initial_scores, current_scores)