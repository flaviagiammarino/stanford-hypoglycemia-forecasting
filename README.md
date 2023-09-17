# Machine learning algorithm for week-ahead hypoglycemia prediction from continuous glucose monitoring data

This algorithm takes as input the patients' continuous glucose monitoring (CGM) data over a given week, 
and outputs the probability that each patient will experience a hypoglycemic episode over the subsequent week. 

The algorithm consists of two components: an unsupervised feature extraction algorithm which uses random convolutional 
kernels to derive a large number of features from the past values of the patients' CGM time series, and a linear classifier which takes 
as input the extracted features and outputs the predicted probability of a future hypoglycemic episode.

The feature extraction algorithm is the MiniRocket [1] algorithm for variable length inputs, and the code is taken directly from 
the [official code repository](https://github.com/angus924/minirocket). The linear classifier is an L1 and L2 regularised logistic regression trained 
with gradient descent in TensorFlow, and the code is provided in this repository.

<p align="center">
    <img src=diagram.png style="width:70%;"/>
</p>

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
    length=70,   # length of the time series, in days
    num=10,      # number of time series
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
    num=10,      # number of time series
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
print(predictions)
# patient                start                  end  predicted_label  predicted_probability  decision_threshold                                                                                                     
# 0        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.217435                 0.9
# 1        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.394081                 0.9
# 2        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.091520                 0.9
# 3        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.007475                 0.9
# 4        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.013187                 0.9
# 5        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.386588                 0.9
# 6        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.045542                 0.9
# 7        2023-09-17 00:00:00  2023-09-23 23:55:00                1               0.999485                 0.9
# 8        2023-09-17 00:00:00  2023-09-23 23:55:00                0               0.002429                 0.9
# 9        2023-09-17 00:00:00  2023-09-23 23:55:00                1               0.996193                 0.9
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
    length=70,   # length of the time series, in days
    num=10,      # number of time series
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
# accuracy           0.800000
# balanced_accuracy  0.733333
# sensitivity        0.600000
# specificity        0.866667
# auc                0.786667
```

## References

[1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2021. MiniRocket: A very fast (almost) deterministic transform for time series classification. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining* (pp. 248-257).

