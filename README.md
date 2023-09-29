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
The inference algorithm takes as input the one-week sequences $`X^{i}_{t}`$ of one or more patients defined as outlined above.
The input sequences $`X^{i}_{t}`$ are fed to the MiniRocket algorithm, which transforms them into 9,996 features $`Z^{i}_{t}`$.
The extracted features $`Z^{i}_{t}`$ are then passed to the linear classifier which outputs the predicted hypoglycemic event probability $`\hat{p}^{i}_{t + 1}`$ for the subsequent week $`t + 1`$.

The predicted binary labels are obtained by comparing the predicted probability $`\hat{p}^{i}_{t + 1}`$ with the decision threshold $`c`$ previously estimated on the training set.
If $`\hat{p}^{i}_{t + 1} > c`$ (resp. $`\hat{p}^{i}_{t + 1} \le c`$) then the model predicts that patient $`i`$ will (resp. will not) experience a hypoglycemic event over the subsequent week $`t + 1`$.

## Dependencies

```bash
pandas==1.5.3
numpy==1.23.5
scipy==1.10.1
numba==0.56.4
statsmodels==0.13.2
scikit-learn==1.2.2
tensorflow==2.12.0
```

## Usage

### Model hyperparameters

The MiniRocket algorithm uses the default hyperparameters recommended by the authors [1] and their values are not exposed in the code.
The remaining hyperparameters are defined as follows:

- `time_worn_threshold`: (`float`, default = 0.7). <br>
The minimum percentage of time that the patient must have worn the CGM device over a given week.


- `blood_glucose_threshold`: (`int`, default = 54). <br>
The blood glucose level below which we detect the onset of hypoglycemia, in mg/dL.


- `episode_duration_threshold`: (`int`, default = 15). <br>
The minimum length of a hypoglycemic event, in minutes.


- `l1_penalty`: (`float`, default = 0.005). <br>
The L1 penalty of the linear classifier.


- `l2_penalty`: (`float`, default = 0.05). <br>
The L2 penalty of the linear classifier.


- `learning_rate`: (`float`, default = 0.00001). <br>
The learning rate used for training the linear classifier.


- `batch_size`: (`int`, default = 32). <br>
The batch size used for training the linear classifier.


- `epochs`: (`int`, default = 1000). <br>
The maximum number of training epochs of the linear classifier.

### Model training
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_training_data

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=280,  # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_training_data(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
)

# train the model
model = Model()

model.fit(
    sequences=sequences,
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.00001,
    batch_size=32,
    epochs=1000,
    verbose=1
)

# save the model
model.save(directory='model')
```
### Model inference
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_inference_data

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=7,    # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_inference_data(
    data=data,
    time_worn_threshold=time_worn_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# generate the model predictions
predictions = model.predict(sequences=sequences)

print(predictions.head(10))
#    patient                start                  end  predicted_label  predicted_probability  decision_threshold
# 0        0  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.094029                0.45
# 1        1  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.119137                0.45
# 2        2  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.046282                0.45
# 3        3  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.090396                0.45
# 4        4  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.126644                0.45
# 5        5  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.486400                0.45
# 6        6  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.077495                0.45
# 7        7  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.990677                0.45
# 8        8  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.083267                0.45
# 9        9  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.999524                0.45
```
### Model evaluation
```python
from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_train_test_data

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=280,  # length of the time series, in days
    num=100,     # number of time series
)

# split the data into training and test sets
training_sequences, test_sequences = get_train_test_data(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
    test_size=0.2,
)

# fit the model to the training set
model = Model()

model.fit(
    sequences=training_sequences,
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.00001,
    batch_size=32,
    epochs=1000,
    verbose=1
)

# evaluate the model on the test set
metrics = model.evaluate(sequences=test_sequences)

print(metrics)
# accuracy           0.942500
# balanced_accuracy  0.918495
# sensitivity        0.878981
# specificity        0.958009
# auc                0.987846
```

## References

[1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2021. MiniRocket: A very fast (almost) deterministic transform for time series classification. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining* (pp. 248-257).

