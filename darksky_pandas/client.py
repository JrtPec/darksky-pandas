import requests
import datetime as dt
from typing import Union, Iterator, Tuple
import pandas as pd
from dateutil import rrule
from tqdm import tqdm
import numpy as np

base_url = 'https://api.darksky.net/forecast'


def get_forecast(api_key: str, lat: float, lng: float, time: dt.datetime=None, session:requests.Session=None,
                 solar:bool=False, units='si', **params) -> dict:
    r = _req_forecast(api_key=api_key, lat=lat, lng=lng, time=time, session=session, solar=solar, units=units, **params)
    return r.json()

def _req_forecast(api_key: str, lat: float, lng: float, time: dt.datetime=None, session:requests.Session=None,
                  solar:bool=False, **params) -> requests.Response:
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
    return r

def datablock_to_dataframe(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
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

        agg = {'temperature': np.mean, 'solarGhi': np.sum}
        agg = {column: agg[column] for column in agg if column in hourly.columns}

        hourly = hourly.resample('d').agg(agg)
        hourly = hourly.round(2)

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

def get_daily_and_hourly_dataframes(api_key: str, lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp,
                        session: requests.Session=None, solar: bool=True, **params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    days = dayset(start=start, end=end)
    day_frames = []
    hour_frames = []
    for date in tqdm(days):
        time = dt.datetime(year=date.year, month=date.month, day=date.day)
        f = get_forecast(api_key=api_key, lat=lat, lng=lng, time=time, session=session, solar=solar, **params)
        day_frame = forecast_to_daily_dataframe(f)
        hour_frame = hourly_to_dataframe(f['hourly']['data'])
        hour_frame = hour_frame.tz_convert(f['timezone'])
        day_frames.append(day_frame)
        hour_frames.append(hour_frame)
    day_df = pd.concat(day_frames, sort=True)
    hour_df = pd.concat(hour_frames, sort=True)

    return day_df, hour_df

class Client:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_daily_dataframe(self, lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp, solar: bool=True,
                            **params) -> pd.DataFrame:
        return get_daily_dataframe(api_key=self.api_key, lat=lat, lng=lng, start=start, end=end, session=self.session,
                                   solar=solar, **params)

    def get_daily_and_hourly_dataframes(self, lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp,
                                        solar: bool=True, **params) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return get_daily_and_hourly_dataframes(api_key=self.api_key, lat=lat, lng=lng, start=start, end=end,
                                               session=self.session, solar=solar, **params)