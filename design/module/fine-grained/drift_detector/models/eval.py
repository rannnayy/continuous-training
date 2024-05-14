#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, roc_auc_score, precision_score, recall_score

def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def evaluate(y_test, y_pred, output_path, model_name):
    stats = []
    target_names = ["not", "drift"]
    labels_names = [False, True]
    stats.append(classification_report(y_test, y_pred, labels=labels_names, target_names=target_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()

    figure_path = os.path.join(output_path, model_name+'.png')
    plt.savefig(figure_path)
    print("===== output figure : " + figure_path)
    plt.close()
    
    # Calculate ROC-AUC and FPR/FNR
    # TN, FP, FN, TP = cm.ravel()
    cm_values = [0 for i in range(4)]
    i = 0
    for row in cm:
        for val in row:
            cm_values[i] = val
            i += 1
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    
    # Calculate Accuracy, Precision, Recall, F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1_sc = f1_score(y_test, y_pred, zero_division=0)
    
    FPR, FNR = round(FP/(FP+TN + 0.1),3), round(FN/(TP+FN  + 0.1),3)
    
    # Append more stats
    stats.append("FPR = "+ str(FPR) + "  (" + str(round(FPR*100,1))+ "%)")
    stats.append("FNR = "+ str(FNR) + "  (" + str(round(FNR*100,1))+ "%)")
    
    stats_path = os.path.join(output_path, model_name+'.stats')
    write_stats(stats_path, "\n".join(stats))