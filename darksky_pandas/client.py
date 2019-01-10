import forecastio
from forecastio import models
import datetime as dt
from typing import Union
import pandas as pd


def load_forecast(key: str, lat: float, lng: float, time: dt.datetime=None,
                  units: str="auto", lang: str="en", lazy: bool=False,
                  callback=None) -> 'Forecast':
    forecast = forecastio.load_forecast(key=key, lat=lat, lng=lng, time=time,
                                        units=units, lang=lang, lazy=lazy,
                                        callback=callback)
    forecast = Forecast(data=forecast.json, response=forecast.response, headers=forecast.http_headers)
    return forecast


class Forecast(models.Forecast):
    def _forcastio_data(self, key: str) -> Union[pd.Series, pd.DataFrame]:
        data = super(Forecast, self)._forcastio_data(key=key)
        if isinstance(data, models.ForecastioDataPoint):
            series = pd.Series(data.d)
            series['time'] = pd.Timestamp.utcfromtimestamp(series['time']).tz_localize('UTC').tz_convert(self.json['timezone'])
            return series
        else:
            series = [pd.Series(d.d) for d in data.data]
            frame = pd.concat(series, axis=1, sort=False).T
            frame['time'] = frame['time'].apply(pd.Timestamp.utcfromtimestamp)
            frame.set_index('time', inplace=True)
            frame = frame.tz_localize('UTC').tz_convert(self.json['timezone'])
            frame.sort_index(inplace=True)
            return frame