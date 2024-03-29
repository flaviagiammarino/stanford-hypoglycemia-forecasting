import os
import logging
import random
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
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
    def fit(self, sequences, reference_length, seed):
        
        # extract the input sequences
        X = np.concatenate([s['X'] for s in sequences], dtype=np.float32)
        
        # extract the lengths of the input sequences
        L = np.array([s['L'] for s in sequences], dtype=np.int32)
        
        # get the parameters
        self.parameters = fit(X=X, L=L, reference_length=reference_length, seed=seed)
    
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
            seed,
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
        set_global_determinism(seed)
        
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
            validation_split=0.3,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    start_from_epoch=100,
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
            sequence_length,
            l1_penalty,
            l2_penalty,
            learning_rate,
            batch_size,
            epochs,
            seed,
            verbose):
        
        # extract the features
        transformer = Transformer()
        transformer.fit(
            sequences=sequences,
            reference_length=sequence_length,
            seed=seed
        )
        features = transformer.transform(sequences=sequences)
        
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
            seed=seed,
            verbose=verbose
        )
        
        # save the objects
        self.transformer = transformer
        self.classifier = classifier
    
    def save(self, directory):
        
        # save the classifier
        self.classifier.model.save(directory)
        
        # save the parameters
        with open(f'{directory}/parameters.json', 'w') as f:
            json.dump({
                'loc': self.classifier.loc.tolist(),
                'scale': self.classifier.scale.tolist(),
                'threshold': self.classifier.threshold,
                'parameters': [self.transformer.parameters[i].tolist() for i in range(len(self.transformer.parameters))]
            }, f)
    
    def load(self, directory):
        
        # load the classifier
        self.classifier = Classifier()
        self.classifier.model = tf.keras.models.load_model(directory)
        
        # load the parameters
        with open(f'{directory}/parameters.json', 'r') as f:
            parameters = json.load(f)

        self.classifier.loc = parameters['loc']
        self.classifier.scale = parameters['scale']
        self.classifier.threshold = parameters['threshold']
        
        self.transformer = Transformer()
        self.transformer.parameters = (
            np.array(parameters['parameters'][0], np.int32),
            np.array(parameters['parameters'][1], np.int32),
            np.array(parameters['parameters'][2], np.float32)
        )
    
    def evaluate(self, sequences):
        
        # extract the features
        features = self.transformer.transform(sequences)
    
        # extract the targets
        targets = np.array([s['Y'] for s in sequences], dtype=np.int32)
        
        # generate the model predictions
        predictions, probabilities = self.classifier.predict(features)
        
        # calculate the model performance metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'balanced_accuracy': balanced_accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions),
            'sensitivity': sensitivity(targets, predictions),
            'specificity': specificity(targets, predictions),
            'f1': f1_score(targets, predictions),
            'auc': roc_auc_score(targets, probabilities)
        }
        
        return metrics
        
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
            'predicted_label': predictions[i],
            'predicted_probability': probabilities[i],
            'decision_threshold': self.classifier.threshold
        } for i in range(len(sequences)))
        
        return results


def get_optimal_threshold(inputs, outputs, model):
    '''
    Find the decision threshold that minimizes the difference between sensitivity and specificity.
    '''
    thresholds = np.linspace(0.01, 0.99, 99)
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
