import os
import logging
import random
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
tf.autograph.set_verbosity(0)

from src.minirocket_variable import fit, transform

class Transformer:
    '''
    Transform the inputs with random convolutional kernels.
    '''
    
    def __init__(self, num_features):
        self.num_features = num_features
    
    def fit(self, sequences):
        
        # extract the input sequences
        X = np.concatenate([s['X'] for s in sequences], dtype=np.float32)
        
        # extract the lengths of the input sequences
        L = np.array([s['L'] for s in sequences], dtype=np.int32)
        
        # get the parameters
        self.parameters = fit(X=X, L=L, num_features=self.num_features, reference_length=np.min(L))
    
    def transform(self, sequences):
        
        # extract the input sequences
        X = np.concatenate([s['X'] for s in sequences], dtype=np.float32)
        
        # extract the lengths of the input sequences
        L = np.array([s['L'] for s in sequences], dtype=np.int32)
        
        # get the features
        return transform(X=X, L=L, parameters=self.parameters)


class Classifier():
    '''
    Fit an L1 and L2 regularised linear classifier to the transformed inputs.
    '''
    
    def fit(self,
            features,
            targets,
            l1_penalty,
            l2_penalty,
            learning_rate,
            batch_size,
            epochs,
            verbose):
        # copy the features and targets
        features = features.copy()
        targets = targets.copy()
        
        # shuffle the features and targets
        features, targets = shuffle(features, targets, random_state=42)
        
        # scale the features
        loc = np.mean(features, axis=0)
        scale = np.std(features, axis=0, ddof=1)
        features = (features - loc) / scale
        
        # build the model
        set_global_determinism(42)
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=1,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_penalty, l2=l2_penalty),
                activation='sigmoid'
            ),
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )
        
        # train the model
        history = model.fit(
            x=features,
            y=targets,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                )
            ],
            verbose=verbose,
        )
        
        # select the threshold
        threshold = get_optimal_threshold(features, targets, model)
        
        # save the objects
        self.history = history
        self.model = model
        self.loc = loc
        self.scale = scale
        self.threshold = threshold
    
    def predict(self, features):
        
        # scale the features
        features = (features - self.loc) / self.scale
        
        # get the predicted probabilities
        probabilities = self.model(features).numpy().flatten()
        
        # get the predicted class labels
        predictions = np.where(probabilities > self.threshold, 1, 0)
        
        return predictions, probabilities


class Model():
    
    def fit(self,
            sequences,
            num_features,
            l1_penalty,
            l2_penalty,
            learning_rate,
            batch_size,
            epochs,
            verbose):
        
        # extract the features
        transformer = Transformer(num_features=num_features)
        transformer.fit(sequences)
        features = transformer.transform(sequences)
        
        # extract the targets
        targets = np.array([s['Y'] for s in sequences], dtype=np.int32)
        
        # fit the classifier
        classifier = Classifier()
        
        classifier.fit(
            features=features,
            targets=targets,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )
        
        # save the objects
        self.transformer = transformer
        self.classifier = classifier
    
    def predict(self, sequences):
        
        # extract the features
        features = self.transformer.transform(sequences)
        
        # generate the model predictions
        predictions, probabilities = self.classifier.predict(features)
        
        # organize the model predictions in a data frame
        results = pd.DataFrame({
            'patient': sequences[i]['patient'],
            'start': sequences[i]['start'],
            'end': sequences[i]['end'],
            'actual_label': sequences[i]['Y'] if 'Y' in sequences[i].keys() else None,
            'predicted_label': predictions[i],
            'predicted_probability': probabilities[i],
            'decision_threshold': self.classifier.threshold
        } for i in range(len(sequences)))
        
        return results


def get_optimal_threshold(inputs, outputs, model):
    '''
    Find the decision threshold that minimizes the difference between sensitivity and specificity.
    '''
    thresholds = np.linspace(0.05, 0.95, 19)
    differences = np.array([])
    for threshold in thresholds:
        predictions = (model(inputs).numpy().flatten() > threshold).astype(int)
        differences = np.append(differences, sensitivity(outputs, predictions) - specificity(outputs, predictions))
    return thresholds[np.argmin(np.abs(differences))]


def sensitivity(y_true, y_pred):
    '''
    Calculate the sensitivity.
    '''
    return recall_score(y_true, y_pred, pos_label=1)


def specificity(y_true, y_pred):
    '''
    Calculate the specificity.
    '''
    return recall_score(y_true, y_pred, pos_label=0)


def set_global_determinism(seed):
    '''
    Fix all sources of randomness.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
