import tensorflow as tf
from xgboost import XGBClassifier
import joblib
import numpy as np

import sys
sys.path.append('../')

# Approach 1: Prediction Model
MODEL_PATH_v0 = {
    'randforest': './dd_model_experiments/models/stored_model/randforest/v0/randforest.joblib',
    'multinomialnb': './dd_model_experiments/models/stored_model/multinomialnb/v0/multinomialnb.joblib',
    'bernoullinb': './dd_model_experiments/models/stored_model/bernoullinb/v0/bernoullinb.joblib',
    'kmeans': "/mnt/extra/continuous-training/design/module/fine-grained/drift_detector/models/stored_model/kmeans/kmeans.joblib",
    'birch': "./dd_model_experiments/models/stored_model/birch/v0/birch.joblib",
    'lof': "./dd_model_experiments/models/stored_model/lof/v0/lof.joblib",
    'xgboost': './dd_model_experiments/models/stored_model/xgboost/v0/xgboost.json',
    'nn': '/mnt/extra/continuous-training/design/module/fine-grained/drift_detector/models/stored_model/nn/v0/nn.tf'
}

# Tune normalizer based on the model itselves later on
NORM_PATH_v0 = {
    'randforest': './dd_model_experiments/models/stored_model/randforest/v0/randforest_norm.joblib',
    'multinomialnb': './dd_model_experiments/models/stored_model/multinomialnb/v0/multinomialnb_norm.joblib',
    'bernoullinb': './dd_model_experiments/models/stored_model/bernoullinb/v0/bernoullinb_norm.joblib',
    'kmeans': "/mnt/extra/continuous-training/design/module/fine-grained/drift_detector/models/stored_model/kmeans/kmeans_norm.joblib",
    'birch': "./dd_model_experiments/models/stored_model/birch/v0/birch_norm.joblib",
    'lof': "./dd_model_experiments/models/stored_model/lof/v0/lof_norm.joblib",
    'xgboost': './dd_model_experiments/models/stored_model/xgboost/v0/xgboost_norm.joblib'
}

MODEL_PATH_v1 = {
    'randforest': './dd_model_experiments/models/stored_model/randforest/v1/randforest.joblib',
    'multinomialnb': './dd_model_experiments/models/stored_model/multinomialnb/v1/multinomialnb.joblib',
    'bernoullinb': './dd_model_experiments/models/stored_model/bernoullinb/v1/bernoullinb.joblib',
    'kmeans': "./dd_model_experiments/models/stored_model/kmeans/v1/kmeans.joblib",
    'birch': "./dd_model_experiments/models/stored_model/birch/v1/birch.joblib",
    'lof': "./dd_model_experiments/models/stored_model/lof/v1/lof.joblib",
    'xgboost': './dd_model_experiments/models/stored_model/xgboost/v1/xgboost.json',
    'nn': './dd_model_experiments/models/stored_model/nn/v1/nn.tf'
}

# Tune normalizer based on the model itselves later on
NORM_PATH_v1 = {
    'randforest': './dd_model_experiments/models/stored_model/randforest/v1/randforest_norm.joblib',
    'multinomialnb': './dd_model_experiments/models/stored_model/multinomialnb/v1/multinomialnb_norm.joblib',
    'bernoullinb': './dd_model_experiments/models/stored_model/bernoullinb/v1/bernoullinb_norm.joblib',
    'kmeans': "./dd_model_experiments/models/stored_model/kmeans/v1/kmeans_norm.joblib",
    'birch': "./dd_model_experiments/models/stored_model/birch/v1/birch_norm.joblib",
    'lof': "./dd_model_experiments/models/stored_model/lof/v1/lof_norm.joblib",
    'xgboost': './dd_model_experiments/models/stored_model/xgboost/v1/xgboost_norm.joblib'
}

MODEL_PATH = {
    'v0': MODEL_PATH_v0,
    'v1': MODEL_PATH_v1
}

NORM_PATH = {
    'v0': NORM_PATH_v0,
    'v1': NORM_PATH_v1
}

def predict_sklearn(model_name, dataset_ver, p_thpt_diff):
    norm = joblib.load(NORM_PATH[dataset_ver][model_name])
    p_thpt_diff = np.array(p_thpt_diff).reshape(1, -1)
    x_norm = norm.transform(p_thpt_diff)
    
    model = joblib.load(MODEL_PATH[dataset_ver][model_name])
    if model_name != 'lof':
        return model.predict(x_norm)[0]
    else:
        return True if model.predict(x_norm)[0] == -1 else False

def predict_xgboost(model_name, dataset_ver, p_thpt_diff):
    norm = joblib.load(NORM_PATH[dataset_ver][model_name])
    x_norm = norm.transform(p_thpt_diff)
    
    model = XGBClassifier()
    model.load_model(MODEL_PATH[dataset_ver][model_name])
    
    return model.predict(x_norm)[0]

def predict_nn(model_name, dataset_ver, p_thpt_diff):
    model = tf.keras.models.load_model(MODEL_PATH[dataset_ver][model_name])
    
    # print((model.predict(np.array(p_thpt_diff)) > 0.5).flatten())
    
    return (model.predict(np.array(p_thpt_diff), verbose=0) > 0.5).flatten()[0]

MODEL_PREDICTOR = {
    'randforest': predict_sklearn,
    'multinomialnb': predict_sklearn,
    'bernoullinb': predict_sklearn,
    'kmeans': predict_sklearn,
    'birch': predict_sklearn,
    'lof': predict_sklearn,
    'xgboost': predict_xgboost,
    'nn': predict_nn
}

def dd_model(model_name, p_thpt_diff):
    dataset_ver = 'v0'
    isDrift = MODEL_PREDICTOR[model_name](model_name, dataset_ver, p_thpt_diff)
    
    return isDrift