import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from scipy.special import softmax

from rocket_functions import generate_kernels, apply_kernels

# Genera kernel convoluzionali casuali
input_length = X_train.shape[1]
num_kernels = 1000
kernels = generate_kernels(input_length, num_kernels)

# Applica i kernel alle serie temporali
features_train = apply_kernels(X_train2, kernels)
features_test = apply_kernels(X_test2, kernels)


# Addestramento del modello supervisionato
model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
model.fit(features_train, y_train)

# Predizione delle anomalie nei dati di test
y_pred = model.predict(features_test)

# Per separare multiclasse o monoclasse
if  len(np.unique(y_test)) > 2:
    y_proba = softmax(model.decision_function(features_test), axis=1)
else:
    y_proba = softmax(model.decision_function(features_test), axis=0)