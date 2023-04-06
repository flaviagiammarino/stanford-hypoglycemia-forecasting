import os
import logging
import random
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
tf.autograph.set_verbosity(0)

from src.minirocket_variable import fit, transform

class MiniRocket():
    '''
    Feature extractor.
    '''
    def fit(self, sequences):
        
        # extract the input sequences
        X = np.concatenate([s['X'] for s in sequences], dtype=np.float32)
    
        # extract the lengths of the input sequences
        L = np.array([s['L'] for s in sequences], dtype=np.int32)
        
        # get the parameters
        self.parameters = fit(X=X, L=L)
    
    def transform(self, sequences):
        
        # extract the input sequences
        X = np.concatenate([s['X'] for s in sequences], dtype=np.float32)
    
        # extract the lengths of the input sequences
        L = np.array([s['L'] for s in sequences], dtype=np.int32)
    
        # get the features
        return transform(X=X, L=L, parameters=self.parameters)


class Classifier():
    '''
    Regularized linear classifier.
    '''
    def fit(self,
            inputs,
            outputs,
            l1_penalty,
            l2_penalty,
            learning_rate,
            batch_size,
            epochs,
            verbose):
        
        # copy the features and targets
        inputs = inputs.copy()
        outputs = outputs.copy()
        
        # scale the features
        mu = np.mean(inputs, axis=0)
        sigma = np.std(inputs, axis=0, ddof=1)
        inputs = (inputs - mu) / sigma

        # train the model
        set_global_determinism(seed=42)
        
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

        history = model.fit(
            x=inputs,
            y=outputs,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )

        # select the threshold
        threshold = get_optimal_threshold(inputs, outputs, model)
        
        # save the objects
        self.history = history
        self.model = model
        self.mu = mu
        self.sigma = sigma
        self.threshold = threshold
    
    def predict(self, inputs):
        
        # scale the features
        inputs = (inputs - self.mu) / self.sigma
        
        # get the predicted probabilities
        probabilities = self.model(inputs).numpy().flatten()
        
        # get the predicted class labels
        predictions = np.where(probabilities >= self.threshold, 1, 0)
        
        return predictions, probabilities


class Model():
    '''
    Combined feature extractor and regularized linear classifier.
    '''
    def fit(self,
            sequences,
            l1_penalty,
            l2_penalty,
            learning_rate,
            batch_size,
            epochs,
            verbose):
        
        # extract the features
        minirocket = MiniRocket()
        minirocket.fit(sequences)
        Z = minirocket.transform(sequences)
    
        # extract the targets
        Y = np.array([s['Y'] for s in sequences], dtype=np.int32)

        # fit the classifier
        classifier = Classifier()

        classifier.fit(
            inputs=Z,
            outputs=Y,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )
        
        # save the objects
        self.minirocket = minirocket
        self.classifier = classifier

    def predict(self, sequences):
        
        # extract the features
        Z = self.minirocket.transform(sequences)
    
        # generate the model predictions
        predictions, probabilities = self.classifier.predict(inputs=Z)
    
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
        predictions = (model(inputs).numpy().flatten() >= threshold).astype(int)
        differences = np.append(differences, sensitivity(outputs, predictions) - specificity(outputs, predictions))
    return thresholds[np.argmin(np.abs(differences))]


def set_global_determinism(seed):
    '''
    Fix all sources of randomness to ensure reproducibility.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


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