import numpy as np
from rocket_functions import generate_kernels, apply_kernels

def detect_anomalies_with_threshold(scores, threshold):
    return (scores > threshold).astype(int)

# Genera kernel convoluzionali casuali
input_length = X_train.shape[1]
num_kernels = 10000
kernels = generate_kernels(input_length, num_kernels)

# Applica i kernel alle serie temporali
features_train = apply_kernels(X_train2, kernels)
features_test = apply_kernels(X_test2, kernels)

# Sintesi delle caratteristiche per esempio
anomaly_scores_train = np.mean(features_train, axis=1)  # Media
anomaly_scores_test = np.mean(features_test, axis=1)  # Media

# Rilevamento delle anomalie
threshold = np.percentile(anomaly_scores_train , 95)
anomaly_labels_train = detect_anomalies_with_threshold(anomaly_scores_train , threshold)
anomaly_labels_test = detect_anomalies_with_threshold(anomaly_scores_test , threshold)