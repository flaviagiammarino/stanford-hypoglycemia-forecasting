# Machine learning model for week-ahead hypoglycemia prediction from continuous glucose monitoring data
![license](https://img.shields.io/github/license/flaviagiammarino/stanford-hypoglycemia-forecasting)
![languages](https://img.shields.io/github/languages/top/flaviagiammarino/stanford-hypoglycemia-forecasting)

## Model description
The model takes as input the patient's continuous glucose monitoring (CGM) readings over a given week, 
and outputs the probability that the patient will experience a hypoglycemic event over the subsequent week. 
The model consists of two components: 
- an unsupervised feature extraction algorithm which uses random convolutional kernels to derive a large number 
of features from the patient's CGM time series; 
- a regularized linear classifier which takes as input the extracted features and outputs 
the patient's predicted hypoglycemic event probability.

The feature extraction algorithm is the MiniRocket [1] algorithm for variable length inputs, 
and the code is taken directly from the [official code repository](https://github.com/angus924/minirocket). 
The linear classifier is an L1 and L2 regularized logistic regression trained with gradient descent in TensorFlow, 
and the code is provided in this repository.

### Model training
The training algorithm takes as input the CGM time series of one or more patients $`i \in \{1, 2, \ldots, N\}`$. 
It then splits the patients' CGM time series into non-overlapping one-week sequences and derives the $`(X^{i}_{t}, y^{i}_{t + 1})`$ training pairs, where

- $`X^{i}_{t}`$ is the time series of CGM readings of patient $`i`$ on week $`t`$ (e.g. 2,016 readings for a patient wearing a 5-minute CGM sensor 100% of the time),
- $`y^{i}_{t + 1}`$ is the binary label of patient $`i`$ on week $`t + 1`$, which is equal to 1 if patient $`i`$ experienced a hypoglycemic event during week $`t + 1`$ and equal to 0 otherwise. 

The input sequences $`X^{i}_{t}`$ are fed to the MiniRocket algorithm which transforms them into 9,996 features $`Z^{i}_{t}`$.
The extracted features $`Z^{i}_{t}`$ are then used together with the binary labels $`y^{i}_{t + 1}`$ for training the linear classifier.

Note that the $`(X^{i}_{t}, y^{i}_{t + 1})`$ training pairs of different patients are pooled together before being fed to the model, 
i.e. the training algorithm fits a unique model for the entire cohort of patients, as opposed to fitting a distinct model for each patient.

In addition to learning the model parameters, the training algorithms also finds the optimal decision threshold $`c`$ on the linear classifier's predicted probabilities, 
which is obtained by minimizing the difference between sensitivity and specificity. 

### Model inference
The inference algorithm takes as input the one-week sequences $`X^{i}_{t}`$ of one or more patients, which are defined as outlined above.
The input sequences $`X^{i}_{t}`$ are fed to the MiniRocket algorithm, which transforms them into 9,996 features $`Z^{i}_{t}`$.
The extracted features $`Z^{i}_{t}`$ are then passed to the linear classifier which outputs the predicted hypoglycemic event probability $`\hat{p}^{i}_{t + 1}`$ for the subsequent week $`t + 1`$.

The predicted binary labels are obtained by comparing the predicted probability $`\hat{p}^{i}_{t + 1}`$ with the decision threshold $`c`$ previously estimated on the training set.
If $`\hat{p}^{i}_{t + 1} > c`$, then the model predicts that patient $`i`$ is likely to experience a hypoglycemic event in the subsequent week $`t + 1`$ ($`y^{i}_{t + 1} = 1`$),
while if $`\hat{p}^{i}_{t + 1} \le c`$, then the model predicts that patient $`i`$ is unlikely to experience a hypoglycemic event in the subsequent week $`t + 1`$ ($`y^{i}_{t + 1} = 0`$). 

## Code

### Dependencies

```bash
pandas==2.0.3
numpy==1.23.5
scipy==1.10.1
numba==0.56.4
statsmodels==0.13.5
scikit-learn==1.2.2
tensorflow==2.13.0
optuna==3.4.0
```

### Hyperparameters

The MiniRocket algorithm uses the default hyperparameters recommended by the authors [1] and their values are not exposed in the code.

The linear classifier has the following hyperparameters:

- `l1_penalty`: (`float`). The L1 penalty.
- `l2_penalty`: (`float`). The L2 penalty.
- `learning_rate`: (`float`). The learning rate used for training.
- `batch_size`: (`int`). The batch size used for training.
- `epochs`: (`int`). The maximum number of training epochs.

Note that the linear classifier is trained with early stopping by monitoring the binary cross-entropy loss on a held-out 20% validation set with a patience of 10 epochs.

The following additional hyperparameters are used for deriving the model's input sequences and output labels:

- `time_worn_threshold`: (`float`, default = 0.7). The minimum percentage of time that the patient must have worn the CGM device over a given week.
- `blood_glucose_threshold`: (`int`, default = 54). The blood glucose level below which we detect the onset of hypoglycemia, in mg/dL.
- `episode_duration_threshold`: (`int`, default = 15). The minimum length of a hypoglycemic event, in minutes.

Note that the one-week periods during which the patient has worn the device for a fraction of time lower than `time_worn_threshold` are not used at any stage, neither for training nor for inference.

### Examples

The examples below show how to use the code for training and inference on a set of patients' CGM time series, which for this purpose are artificially generated.

#### Model training example
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_labelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_labelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
)

# train the model
model = Model()

model.fit(
    sequences=sequences,
    l1_penalty=0.01,
    l2_penalty=0.01,
    learning_rate=0.001,
    batch_size=512,
    epochs=1000,
    verbose=1
)

# save the model
model.save(directory='model')
```
#### Model inference example
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_unlabelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=7,    # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_unlabelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# generate the model predictions
predictions = model.predict(sequences=sequences)
print(predictions.head(10))
```
#### Model evaluation example
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_labelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_labelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# evaluate the model
metrics = model.evaluate(sequences=sequences)
print(metrics)
```
#### Hyperparameter tuning example
```python
from src.model import Model, tune_hyperparameters
from src.simulation import simulate_patients
from src.utils import get_train_test_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# split the data into training and test sequences
training_sequences, test_sequences = get_train_test_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
    test_size=0.3,
)

# find the best hyperparameters
parameters, score = tune_hyperparameters(
    sequences=training_sequences,
    n_splits=3,
    n_trials=5,
)

# fit the model to the training set
model = Model()

model.fit(
    sequences=training_sequences,
    l1_penalty=parameters['l1_penalty'],
    l2_penalty=parameters['l2_penalty'],
    learning_rate=parameters['learning_rate'],
    batch_size=parameters['batch_size'],
    epochs=parameters['epochs'],
    verbose=0
)

# evaluate the model on the test set
metrics = model.evaluate(sequences=test_sequences)
print(metrics)
```
## References

[1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2021. MiniRocket: A very fast (almost) deterministic transform for time series classification. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining* (pp. 248-257).

