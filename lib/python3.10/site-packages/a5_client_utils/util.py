from datetime import datetime, timedelta
from typing import Union
import dateutil
import pytz
import logging
from pandas import DatetimeIndex, date_range, DateOffset

def tryParseAndLocalizeDate(
        date_string : Union[str,float,datetime],
        timezone : str='America/Argentina/Buenos_Aires'
    ) -> datetime:
    """
    Datetime parser. If duration is provided, computes date relative to now.

    Parameters:
    -----------
    date_string : str or float or datetime.datetime
        For absolute date: ISO-8601 datetime string or datetime.datetime.
        For relative date: dict (duration key-values) or float (decimal number of days)
    
    timezone : str
        Time zone string identifier. Default: America/Argentina/Buenos_Aires
    
    Returns:
    --------
    datetime.datetime

    Examples:
    ---------
    ``` 
    tryParseAndLocalizeDate("2024-01-01T03:00:00.000Z")
    tryParseAndLocalizeDate(1.5)
    tryParseAndLocalizeDate({"days":1, "hours": 12}, timezone = "Africa/Casablanca")
    ```
    """
    
    date = dateutil.parser.isoparse(date_string) if isinstance(date_string,str) else date_string
    is_from_interval = False
    if isinstance(date,dict):
        date = datetime.now() + dateutil.relativedelta.relativedelta(**date)
        is_from_interval = True
    elif isinstance(date,(int,float)):
        date = datetime.now() + dateutil.relativedelta.relativedelta(days=date)
        is_from_interval = True
    if date.tzinfo is None or date.tzinfo.utcoffset(date) is None:
        try:
            date = pytz.timezone(timezone).localize(date)
        except pytz.exceptions.NonExistentTimeError:
            logging.warning("NonexistentTimeError: %s" % str(date))
            return None
    else:
        date = date.astimezone(pytz.timezone(timezone))
    return date # , is_from_interval

def createDatetimeSequence(
    datetime_index : DatetimeIndex=None, 
    timeInterval  = timedelta(days=1), 
    timestart = None, 
    timeend = None, 
    timeOffset = None
    ) -> DatetimeIndex:
    #Fechas desde timestart a timeend con un paso de timeInterval
    #data: dataframe con index tipo datetime64[ns, America/Argentina/Buenos_Aires]
    #timeOffset sólo para timeInterval n days
    if datetime_index is None and (timestart is None or timeend is None):
        raise Exception("Missing datetime_index or timestart+timeend")
    timestart = timestart if timestart is not None else datetime_index.min()
    timestart = roundDate(timestart,timeInterval,timeOffset,"up")
    timeend = timeend if timeend  is not None else datetime_index.max()
    timeend = roundDate(timeend,timeInterval,timeOffset,"down")
    return date_range(start=timestart, end=timeend, freq=DateOffset(days=timeInterval.days, hours=timeInterval.seconds // 3600, minutes = (timeInterval.seconds // 60) % 60))

def roundDate(date : datetime,timeInterval : timedelta,timeOffset : timedelta=None, to="up") -> datetime:
    date_0 = tryParseAndLocalizeDate(datetime.combine(date.date(),datetime.min.time()))
    if timeOffset is not None:
        date_0 = date_0 + timeOffset 
    while date_0 < date:
        date_0 = date_0 + timeInterval
    if to == "up":
        return date_0
    else:
        return date_0 - timeInterval

def interval2timedelta(interval : Union[dict,float,timedelta]):
    """Parses duration dict or number of days into datetime.timedelta object
    
    Parameters:
    -----------
    interval : dict or float (decimal number of days) or datetime.timedelta
        If dict, allowed keys are:
        - days
        - seconds
        - microseconds
        - minutes
        - hours
        - weeks
    
    Returns:
    --------
    duration : datetime.timedelta

    Examples:

    ```
    interval2timedelta({"hours":1, "minutes": 30})
    interval2timedelta(1.5/24)
    ```
    """
    if isinstance(interval,timedelta):
        return interval
    if isinstance(interval,(float,int)):
        return timedelta(days=interval)
    days = 0
    seconds = 0
    microseconds = 0
    milliseconds = 0
    minutes = 0
    hours = 0
    weeks = 0
    for k in interval:
        if k == "milliseconds" or k == "millisecond":
            milliseconds = interval[k]
        elif k == "seconds" or k == "second":
            seconds = interval[k]
        elif k == "minutes" or k == "minute":
            minutes = interval[k]
        elif k == "hours" or k == "hour":
            hours = interval[k]
        elif k == "days" or k == "day":
            days = interval[k]
        elif k == "weeks" or k == "week":
            weeks = interval[k] * 86400 * 7
    return timedelta(days=days, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks)
