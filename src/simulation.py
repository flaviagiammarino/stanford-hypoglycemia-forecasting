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

        'datetime': pandas.datetime.
            Timestamp.

        'glucose': float.
            Blood glucose level.
    '''
    
    # fix the random seed
    np.random.seed(id)
    
    # generate the timestamps
    ts = pd.date_range(
        start=pd.Timestamp(datetime.date.today()) - pd.Timedelta(days=length),
        end=pd.Timestamp(datetime.date.today()) - pd.Timedelta(minutes=freq),
        freq=f'{freq}T'
    )
    
    # generate the baseline blood glucose level
    bg = np.random.uniform(low=70, high=180)
    bg += sm.tsa.arma_generate_sample(
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
    bg[ts.to_series().isin(simulate_timestamps(hours=[6, 7, 8]))] = np.random.uniform(low=180, high=400, size=length)
    bg[ts.to_series().isin(simulate_timestamps(hours=[12, 13, 14]))] = np.random.uniform(low=180, high=400, size=length)
    bg[ts.to_series().isin(simulate_timestamps(hours=[18, 19, 20]))] = np.random.uniform(low=180, high=400, size=length)
    
    # add some downward spikes after mealtimes
    bg[ts.to_series().isin(simulate_timestamps(hours=[9, 10, 11]))] = np.random.uniform(low=20, high=70, size=length)
    bg[ts.to_series().isin(simulate_timestamps(hours=[15, 16, 17]))] = np.random.uniform(low=20, high=70, size=length)
    bg[ts.to_series().isin(simulate_timestamps(hours=[21, 22, 23]))] = np.random.uniform(low=20, high=70, size=length)
    
    # smooth the spikes
    bg = gaussian_filter1d(input=bg, sigma=3)
    
    # add some missing values
    bg[np.random.randint(low=0, high=len(ts), size=int(0.1 * len(ts)))] = np.nan
    
    return pd.DataFrame({'id': id, 'datetime': ts, 'glucose': bg})


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

        'datetime': pandas.datetime.
            Timestamp.

        'glucose': float.
            Blood glucose level.
    '''
    
    # generate the data
    data = pd.concat([simulate_patient(id, freq, length) for id in range(num)], axis=0)
    
    # reshape the data
    data = data.set_index('datetime').groupby(by='id')['glucose'].resample(f'{freq}T').last().reset_index()
    data = data.pivot(index='datetime', columns=['id'], values=['glucose'])
    data.columns = data.columns.get_level_values(level='id')
    
    return data
