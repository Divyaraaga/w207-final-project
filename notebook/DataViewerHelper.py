import plotly
import plotly.plotly as py
import pandas as pd
from plotly.offline import plot, iplot, plot_mpl, init_notebook_mode
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

PICKLE_PATH = '../pickles'
init_notebook_mode(connected=True)

def get_data(data):
    df = pd.read_pickle(f'{PICKLE_PATH}/enhanced_{data}.pkl')
    return df


def chart(title, df):
    """
    Chart helper for predictions
    """
    layout = dict(title=title,
                  xaxis=dict(
                      rangeselector=dict(
                          buttons=list([
                              dict(step='all'),
                              dict(count=24,
                                   label='2yr',
                                   step='month',
                                   stepmode='backward'),
                              dict(count=12,
                                   label='1yr',
                                   step='month',
                                   stepmode='backward'),
                              dict(count=6,
                                   label='6m',
                                   step='month',
                                   stepmode='backward'),
                              dict(count=1,
                                   label='1m',
                                   step='month',
                                   stepmode='backward')
                          ])
                      ),
                      rangeslider=dict(
                          visible=True
                      ),
                      type='date'
                  )
                  )

    fig = go.Figure(data=[
        {
            'x': df.index,
            'y': df["temperature"],
            'name': "Temperature"
        }], layout=layout)

    return fig