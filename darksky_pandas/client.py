import requests
import datetime as dt
from typing import Union, Iterator
import pandas as pd
from dateutil import rrule
from tqdm import tqdm
import numpy as np

base_url = 'https://api.darksky.net/forecast'


def get_forecast(api_key: str, lat: float, lng: float, time: dt.datetime=None, session:requests.Session=None,
                 solar:bool=False, **params) -> dict:
    url = f'{base_url}/{api_key}/{lat},{lng}'
    if time is not None:
        time_str = time.replace(microsecond=0).isoformat()
        url = f'{url},{time_str}'

    if solar:
        url = f'{url}?&solar'

    if session:
        r = session.get(url=url, params=params)
    else:
        r = requests.get(url=url, params=params)
    r.raise_for_status()
    return r.json()

def datablock_to_dataframe(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df.tz_localize('UTC')
    return df

def hourly_to_dataframe(d: dict) -> pd.DataFrame:
    df = datablock_to_dataframe(d)

    if 'solar' in df.columns:
        solar = df['solar'].dropna().apply(pd.Series)
        solar['azimuth'] = solar['azimuth'].apply(lambda x: (x + 90) % 360)

        solar.columns = [f'solar{name.title()}' for name in solar.columns]
        df = df.join(solar)
        df.drop('solar', axis=1, inplace=True)

    return df

def forecast_to_daily_dataframe(d: dict) -> pd.DataFrame:
    df = datablock_to_dataframe(d['daily']['data'])
    df = df.tz_convert(d['timezone'])

    if 'hourly' in d:
        hourly = hourly_to_dataframe(d['hourly']['data'])
        hourly = hourly.tz_convert(d['timezone'])

        assert 'temperature' in hourly.columns
        assert 'solarGhi' in hourly.columns

        hourly = hourly.resample('d').agg({'temperature': np.mean, 'solarGhi': np.sum})

        df = df.join(hourly)
    return df

def dayset(start: Union[dt.datetime, pd.Timestamp], end: Union[dt.datetime, pd.Timestamp]) -> Iterator[dt.datetime]:
    """Takes a start and end date and returns a set containing all dates between start and end"""
    res = []
    for day in rrule.rrule(rrule.DAILY, dtstart=start, until=end):
        res.append(day.date())
    return sorted(set(res))

def get_daily_dataframe(api_key: str, lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp,
                        session: requests.Session=None, solar: bool=True, **params) -> pd.DataFrame:
    days = dayset(start=start, end=end)
    def gen_day_frames(_days):
        for date in tqdm(_days):
            time = dt.datetime(year=date.year, month=date.month, day=date.day)
            f = get_forecast(api_key=api_key, lat=lat, lng=lng, time=time, session=session, solar=solar, **params)
            frame = forecast_to_daily_dataframe(f)
            yield frame
    frames = gen_day_frames(days)
    df = pd.concat(frames, sort=True)
    return df

class Client:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_daily_dataframe(self, lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp, solar: bool=True,
                            **params) -> pd.DataFrame:
        return get_daily_dataframe(api_key=self.api_key, lat=lat, lng=lng, start=start, end=end, session=self.session,
                                   solar=solar, **params)