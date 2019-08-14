# General libraries
import warnings
import pandas as pd

from pandas.tseries.offsets import MonthEnd
from pandas import Series
import datetime as dt
import numpy as np
from pandas import datetime
import socket
from pathlib import Path
import os

# Visualisation libraries
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Plotly
import plotly
import plotly.plotly as py
from plotly.offline import plot, iplot, plot_mpl, init_notebook_mode
import plotly.graph_objs as go
import cufflinks as cf
from IPython.display import display, Markdown
import ipywidgets as widgets
import nbconvert

# Set credentials for saving plotly Images
plotly.tools.set_credentials_file(username = 'sstorey', api_key = 's1LlKULmU1VXGH8kLpKC')

# DECISION TREE Library
from sklearn import tree

# RANDOM FOREST REGRESSOR Library
from sklearn.ensemble import RandomForestRegressor

# keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam

# ARIMA
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

# Mean Squared error library
from sklearn.metrics import mean_squared_error

# Chart handling
warnings.filterwarnings('ignore')
mpl.style.use('seaborn')

init_notebook_mode(connected = True)

# Path to read and write data files
DATA_PATH = '../data'

# Path to read and write pickle files
PICKLE_PATH = '../pickles'

# Path to save EDA images
EDA_IMAGE_PATH = '../Images/EDA'

# Path to save experiment results
EXP_PATH = '../experiment_results'

# helper functions
def charter_helper_fitted(title, predictions):
    """
    Fitted charts
    """
    layout = dict(title = title, xaxis = dict( rangeselector = dict( buttons = list([ dict(step = 'all'), dict(count = 1, label = '1m', step = 'month', stepmode = 'backward')])), rangeslider = dict(visible = True),type = 'date'))
    fig = go.Figure(data = [{'x': predictions.index,'y': predictions[col],'name': col} for col in predictions.columns ], layout = layout)
    return fig


def get_length_results():
    """
    """
    results_file = Path(f'{PICKLE_PATH}/results.pkl')
    length_of_results = 0
    if (results_file.exists()):
        results_df = pd.read_pickle(results_file)
        length_of_results = results_df['RUN_ID'].max()

    return length_of_results


def create_results_perrun():
    """
    Create directory
    for each set of results
    """
    id = get_length_results() + 1
    EXP_DIR = f'{EXP_PATH}/RUN-{id}'
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
    return EXP_DIR, id


def display_results(results_df, city = 'all', feature_type = 'all', chart_type = None):
    """
    Display result charts
    """
    if (city != 'all'):
        results_df = results_df[results_df['CITY'] == city]

    if (feature_type != 'all'):
        results_df = results_df[results_df['FEATURE_TYPE'] == feature_type]

    for index,row in results_df.iterrows():

        id = row['RUN_ID']
        title = '### Experiment #{0} - run by {1} on {2}'.format(id, row['HOST_MACHINE'],row['DATETIME'])
        display(Markdown(title))
        RUN_PATH = f"RUN-{id}"

        if (chart_type==None or chart_type=="actual"):
            actual_fitted = f"![]({EXP_PATH}/" + RUN_PATH + "/Daily_actual_vs_predict.png)"
            display(Markdown(actual_fitted))
        if (chart_type == None or chart_type == "predict"):
            predicted = f"![]({EXP_PATH}/" + RUN_PATH + "/Daily_predict.png)"
            display(Markdown(predicted))


def charter_helper_prediction(title, X_train, Y_train, X_test, Y_test, future_forecast):
    """
    Chart helper for predictions
    """
    layout = dict(title=title,
        xaxis = dict(
            rangeselector = dict(
                buttons = list([
                    dict(step = 'all'),
                    dict(count = 6,
                        label = '6m',
                        step = 'month',
                        stepmode = 'backward'),
                    dict(count = 1,
                        label = '1m',
                        step = 'month',
                        stepmode = 'backward')
                ])
            ),
            rangeslider = dict(
                visible = True
            ),
            type='date'
        )
    )

    date_index = X_train.shape[0] - 365

    fig = go.Figure(data = [
        {
            'x': X_train.index[date_index: ],
            'y': Y_train["temperature"][date_index: ],
            'name': "Train temperature"
        },
        {
            'x': X_test.index,
            'y': Y_test["temperature"],
            'name': "Actual temperature"
        },
        {
            'x': future_forecast.index,
            'y': future_forecast["Fitted"],
            'name': "Predicted temperature"
        }], layout = layout)

    return fig


def update_results_function(id, model_name, city, feature_type,model_parameters, model_results, mse):
    """
    Update results df / pickle with info from experiment
    """
    results_file = Path(f'{PICKLE_PATH}/results.pkl')

    # Make run id
    RUN_ID = 'RUN-' + str(get_length_results() + 1)

    # Create / Read results df from pickle
    if not(results_file.exists()):
        columns = ['RUN_ID', "DATETIME", "MODEL_NAME", "CITY",
                "FEATURE_TYPE", "HOST_MACHINE", "MODEL_PARAMETERS", "MODEL_RESULTS",
                "MEAN_SQUARED_ERROR"
                ]
        results = pd.DataFrame(columns = columns)
        results.set_index('RUN_ID')
    else:
        results = pd.read_pickle(results_file)

    # If run exists, update
    if (results['RUN_ID'] == id).any():
        row = results[results['RUN_ID'] == id]
        row.DATETIME = datetime.now()
        row.MODEL_NAME = model_name
        row.CITY = city
        row.FEATURE_TYPE = feature_type
        row.HOST_MACHINE = socket.gethostname()
        row.MODEL_PARAMETERS = model_parameters
        row.MODEL_RESULTS = model_results
        row.MEAN_SQUARED_ERROR = mse
    else:
        results = results.append(
        {
            "RUN_ID": id,
            "DATETIME": datetime.now(),
            "MODEL_NAME": model_name,
            "CITY": city,
            "FEATURE_TYPE": feature_type,
            "HOST_MACHINE": socket.gethostname(),
            "MODEL_PARAMETERS": model_parameters,
            "MODEL_RESULTS": model_results,
            "MEAN_SQUARED_ERROR": mse
        }, ignore_index=True)

    # Save pickle
    results.to_pickle(results_file)

    return results


def get_results(city='all', top_hm_results='all', display=False):
    """
    Get results dataframe
    """
    results_file = Path(f'{PICKLE_PATH}/results.pkl')
    if (results_file.exists()):
        results_df = pd.read_pickle(results_file)
        results_df.index = range(1, len(results_df.index) + 1)
        if (city != 'all'):
            results_df = results_df[results_df['CITY'] == city]
        if (top_hm_results != 'all'):
            results_df = results_df.sort_values('MEAN_SQUARED_ERROR', axis=0, ascending=True)
            results_df = results_df[:top_hm_results]
            results_df.index = range(1, len(results_df.index) + 1)

        return results_df


def show_feature_importances(features, importances, top_hm_features=100):
    """
    Show table of feature importances
    """
    features_importances_df = pd.DataFrame()
    features_importances_df["Features"] = features
    features_importances_df["Feature Importances in (%)"] = importances * 100
    features_importances_df = features_importances_df.sort_values('Feature Importances in (%)', axis=0, ascending=False)
    return features_importances_df[:top_hm_features]


def get_feature_importances(results_df, city='all'):
    """
    Get features from runs included in the results_df ( filter by city if reqd )
    """
    if (city != 'all'):
        results_df = results_df[results_df['CITY'] == city]

    # Collect all features into single df
    all_features = pd.DataFrame(columns=('RUN_ID','FEATURE','IMPORTANCE','ORDER','MODEL_NAME','FEATURE_TYPE'))

    # For each run
    for index,row in results_df.iterrows():
        id = row['RUN_ID']
        model_name = row['MODEL_NAME']
        feature_type = row['FEATURE_TYPE']

        # Make path and open feature file from RUN_ID ( if it exists, else skip )
        features_file = Path(f'{EXP_PATH}/RUN-{id}/feature_importances.csv')

        if (features_file.exists()):

            # Open file
            features_df = pd.read_csv(features_file)

            # Extract features
            for f_index, f_row in features_df.iterrows():
                all_features = all_features.append([{'RUN_ID':id,'FEATURE':f_row[1],'IMPORTANCE':f_row[2],'ORDER':f_row[0],'MODEL_NAME':model_name, 'FEATURE_TYPE':feature_type}],ignore_index=True)

    return all_features


def create_boxplot_traces_for_features(features_df):
    """
    Create traces set necessary for boxplot ( for features )
    """
    traces = list()

    for feature in features_df['FEATURE'].unique():
        if str(feature) == 'nan':
            traces.append(go.Box(y=features_df[pd.isnull(features_df['FEATURE'])]['IMPORTANCE'],
                                                name=str(feature)))
        else:
            traces.append(go.Box(y=features_df[features_df['FEATURE'] == feature]['IMPORTANCE'],
                                                name=str(feature)))

    return traces



def create_boxplot_traces_for_results(features_df, column, value,title="Box plot"):
    """
    Create traces set necessary for boxplot ( for mse )
    """
    traces = list()

    for feature in features_df[column].unique():
        if str(feature) == 'nan':
            traces.append(go.Box(y=features_df[pd.isnull(features_df[column])][value],
                                                name=str(feature)))
        else:
            traces.append(go.Box(y=features_df[features_df[column] == feature][value],
                                                name=str(feature)))

    layout = go.Layout(
        title = title
    )

    fig = go.Figure(data=traces,layout=layout)
    iplot(fig)


