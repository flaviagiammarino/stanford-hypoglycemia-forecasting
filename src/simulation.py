import warnings
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings('ignore')

def simulate_patient(id, freq, length):
    '''
    Simulate a single patient's CGM time series.
    
    Parameters:
    ----------------------------------
    id: int.
        Patient id.

    freq: int.
        Frequency of the time series (in minutes).
    
    length: int.
        Length of the time series (in days).

    Returns:
    ----------------------------------
    pd.DataFrame.
        Simulated CGM time series for a single patient.
        pandas.DataFrame with the following columns:

        'id': int.
            Patient id.

        'ts': pandas.datetime.
            Timestamp.

        'gl': float.
            Glucose level.
    '''
    
    # fix the random seed
    np.random.seed(id)
    
    # generate the timestamps
    ts = pd.date_range(
        start=pd.Timestamp(datetime.date.today()) - pd.Timedelta(days=length),
        end=pd.Timestamp(datetime.date.today()) - pd.Timedelta(minutes=freq),
        freq=f'{freq}T'
    )
    
    # generate the baseline glucose level
    gl = np.random.uniform(low=70, high=180)
    gl += sm.tsa.arma_generate_sample(
        ar=np.r_[1, - np.array([.75, -.25])],
        ma=np.r_[1, np.array([.65, .35])],
        scale=10.,
        nsample=len(ts)
    )
    
    def simulate_timestamps(hours):
        '''
        Generate random timestamps within certain hours on each day.

        Parameters:
        ----------------------------------
        hours: list of int.
            Hours within which the timestamps should fall.

        Returns:
        ----------------------------------
        list of datetime.datetime.
        '''
        
        fn = lambda date: datetime.datetime.combine(
            date=date,
            time=datetime.time(
                hour=np.random.choice(a=hours),
                minute=np.random.choice(a=np.arange(start=freq, stop=60, step=freq))
            )
        )
        return [fn(date) for date in ts.to_series().dt.date.unique()]
    
    # add some upward spikes around mealtimes
    gl[ts.to_series().isin(simulate_timestamps(hours=[6, 7, 8]))] = np.random.uniform(low=180, high=400, size=length)
    gl[ts.to_series().isin(simulate_timestamps(hours=[12, 13, 14]))] = np.random.uniform(low=180, high=400, size=length)
    gl[ts.to_series().isin(simulate_timestamps(hours=[18, 19, 20]))] = np.random.uniform(low=180, high=400, size=length)
    
    # add some downward spikes after mealtimes
    gl[ts.to_series().isin(simulate_timestamps(hours=[9, 10, 11]))] = np.random.uniform(low=20, high=70, size=length)
    gl[ts.to_series().isin(simulate_timestamps(hours=[15, 16, 17]))] = np.random.uniform(low=20, high=70, size=length)
    gl[ts.to_series().isin(simulate_timestamps(hours=[21, 22, 23]))] = np.random.uniform(low=20, high=70, size=length)
    
    # smooth the spikes
    gl = gaussian_filter1d(input=gl, sigma=3)
    
    # add some missing values
    gl[np.random.randint(low=0, high=len(ts), size=int(0.1 * len(ts)))] = np.nan
    
    return pd.DataFrame({'id': id, 'ts': ts, 'gl': gl})


def simulate_patients(freq, length, num):
    '''
    Simulate multiple patients' CGM time series.
    
    Parameters:
    ----------------------------------
    freq: int.
        Frequency of the time series (in minutes).

    length: int.
        Length of the time series (in days).
        
    num: int.
        Number of time series (i.e. number of patients).
        
    Returns:
    ----------------------------------
    pd.DataFrame.
        Simulated CGM time series for multiple patients.
        pandas.DataFrame with the following columns:

        'id': int.
            Patient id.

        'ts': pandas.datetime.
            Timestamp.

        'gl': float.
            Glucose level.
    '''
    
    # generate the data
    data = pd.concat([simulate_patient(id, freq, length) for id in range(num)], axis=0)

    # sort the data
    data = data.set_index('ts').groupby(by='id')['gl'].resample(f'{freq}T').last().reset_index()
    
    return data
