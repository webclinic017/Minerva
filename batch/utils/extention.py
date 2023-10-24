import pandas as pd

from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

start_session = pd.Timestamp('2017-01-01', tz='utc')
end_session = pd.Timestamp('2019-06-30', tz='utc')

# register the bundle 
register(
    'quandl', # name we select for the bundle
    csvdir_equities(
        ['Data'], # name of the directory as specified above (named after data frequency)
        './codes/jupyter/Minerva/Data/AAPL.csv', # path to directory containing the data
    ),
    calendar_name='NYSE',  # US equities
    start_session=start_session,
    end_session=end_session
)