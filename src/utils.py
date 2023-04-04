import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_episodes_duration(blood_glucose,
                          blood_glucose_threshold):
    '''
    Get the durations of the hypoglycemic episodes of a given patient, in number of minutes.
    '''
    
    # get the frequency of the data, in number of minutes
    freq = blood_glucose.index.to_series().diff().mode()[0].total_seconds() // 60
    
    # get the episodes indicators
    x = np.logical_and(blood_glucose <= blood_glucose_threshold, ~np.isnan(blood_glucose)).astype(int)
    
    # get the episodes durations
    return np.array([len(x) * freq for x in [list(group) for key, group in itertools.groupby(x)] if 1 in x], dtype=np.int32)


def get_sequence_label(blood_glucose,
                       episode_duration_threshold,
                       blood_glucose_threshold):
    '''
    Label a patient's one-week subsequence as 1 if the patient had a severe hypoglycemic
    episode during the week and as 0 otherwise.
    '''
    
    # get the durations of the patient's hypoglycemic episodes over the given week
    d = get_episodes_duration(blood_glucose, blood_glucose_threshold)
    
    # check if the patient had any hypoglycemic episodes over the given week
    if len(d) > 0:
        
        # check if the length of any hypoglycemic episodes was above the threshold
        if np.max(d) >= episode_duration_threshold:
            
            # if the patient had a hypoglycemic episode with length above the threshold
            # label the sequence as 1
            return 1
        
    # if the patient didn't have any hypoglycemic episode, or if the episodes length
    # was below the threshold, label the sequence as 0
        else:
            return 0
    else:
        return 0
    

def get_sequences(data,
                  time_worn_threshold,
                  blood_glucose_threshold,
                  episode_duration_threshold):
    '''
    Get the input sequences `X` and target values `Y`, in a list of dictionaries called `sequences`
    with the following items:
    
        patient: str.
            The patient id.
            
        start: pd.datetime.
            The first timestamp of the subsequent week.
        
        end: pd.datetime.
            The last timestamp of the subsequent week.
            
        X: np.ndarray.
            The patient's blood glucose measurements over the current week.
        
        L: np.ndarray.
            The patient's number of non-missing blood glucose measurements over the current week.
        
        Y: np.ndarray.
            A binary class label indicating whether the patient had a severe hypoglycemic episode
            over the subsequent week (Y = 1) or not (Y = 0).
    '''
    
    # get the frequency of the data, in number of minutes
    minutes = int(data.index.to_series().diff().mode()[0].total_seconds() // 60)
    
    # calculate the number of timestamps in one week
    sequence_length = int(7 * 24 * 60 // minutes)
    
    sequences = []
    
    # loop across the patients in the dataset
    for patient in data.columns:
    
        # loop across the weeks in the dataset
        for t in range(sequence_length, (data.shape[0] // sequence_length) * sequence_length, sequence_length):
            
            # extract the patient's data over the current week
            X = data[patient].iloc[t - sequence_length: t]
    
            # extract the patient's data over the subsequent week
            Y = data[patient].iloc[t: t + sequence_length]
    
            # check if the patient has worn the device for a sufficient time over both weeks
            if pd.notna(X).mean() >= time_worn_threshold and pd.notna(Y).mean() >= time_worn_threshold:
          
                # drop the missing values from the input sequence
                X = X.dropna().to_list()
        
                # calculate the length of the input sequence
                L = len(X)
        
                # derive the class label
                Y = get_sequence_label(Y, episode_duration_threshold, blood_glucose_threshold)
                
                sequences.append({
                    'patient': patient,
                    'start': data.index[t - 1] + pd.Timedelta(minutes=minutes),
                    'end': data.index[t - 1] + pd.Timedelta(days=7),
                    'X': X,
                    'L': L,
                    'Y': Y,
                })
        
    return sequences

def split_sequences(sequences, test_size):
    '''
    Split the sequences (provided as a list of dictionaries as described in the docstring
    of the `get_sequences()` function) into stratified training and test sets.
    '''
    
    training_sequences, test_sequences = train_test_split(
        sequences,
        stratify=[s['Y'] for s in sequences],
        test_size=test_size,
        random_state=42
    )
    
    return training_sequences, test_sequences
