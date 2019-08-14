
# coding: utf-8

# # Time Series to observe DAILY temperature variations 
# ## Daily temperature prediction using RNN

# Moving from predictive Machine Learning classifier to Unpredicitve Neural Nets, we use Sequential ** Recurrent Neural Net (RNN)** in the notebook below.
# 
# You need to ensure that you have the right environment installed on top of your python3 to run Keras and Tensorflow. Two libraries needed to successfully run (RNN).
# 
# Just like in the other models, we begin by loading all necessary libraries and paths to read the "pickles" as well as store image for the graph towards the end of our code. The pickles are read and the data is fed into an RNN model.
# Finally, we have two graphs showing the DT results vs. the fitted model as well as predicted results vs. actuals and test data

# Here we are importing the train and test Data from pickle files created through the EDA file

# In[274]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('run', 'helper_functions.py')
get_ipython().run_line_magic('matplotlib', 'inline')


# Create a folder for every run of the RNN to store images

# In[275]:


city='Miami' # New_York Atlanta Boston Dallas Houston Miami
analysis_type = 'Enhanced_Signals' # Basic, Inc_Signals, Enhanced_Signals


# In[276]:


EXPERIMENT_DIR, EXPERIMENT_ID  = create_results_perrun()
print("Path of the results directory",EXPERIMENT_DIR )


# In[277]:


#EXPERIMENT_DIR = '../experiment_results/RUN-37'
#EXPERIMENT_ID = 37


# In[278]:


X_train = pd.read_pickle(f'{PICKLE_PATH}/X_train_{city}_{analysis_type}.pkl')
Y_train = pd.read_pickle(f'{PICKLE_PATH}/Y_train_{city}_{analysis_type}.pkl')

X_test  = pd.read_pickle(f'{PICKLE_PATH}/X_test_{city}_{analysis_type}.pkl')
Y_test  = pd.read_pickle(f'{PICKLE_PATH}/Y_test_{city}_{analysis_type}.pkl')

print("Shape of Training Dataset " , X_train.shape)
print("Shape of Testing Dataset " , X_test.shape)


# In[279]:


# Function to fit a sequential rnn with loss estimate being mean squared error
def train_model(X_train, y_train, X_test, y_test, epochs):
    model = Sequential(
        [
            Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(10, activation="relu"),
            Dense(10, activation="relu"),
            Dense(1, activation="linear")
        ]
    )
    model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")
    
    history = model.fit(X_train, y_train, epochs=epochs, shuffle=False)
    return model, history


# In[280]:


# Function to fit a sequential rnn with epochs = 50
epochs = 50
model_encoded, encoded_hist = train_model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    epochs=epochs
)


# In[281]:


# Run the model on the training dataset
Y_train_pred = model_encoded.predict(X_train)
# Calculate mean squared error for the predicted values
mse_train = mean_squared_error(Y_train, model_encoded.predict(X_train))
print('Mean Squared Error for the training dataset: %.3f' % mse_train)  


# In[282]:


# Run the model on the testing dataset
Y_test_pred = model_encoded.predict(X_test)
# Calculate mean squared error for the test vs predicted values
mse_test = mean_squared_error(Y_test, model_encoded.predict(X_test))
print('Mean Squared Error for the testing dataset: %.3f' % mse_test) 


# In[283]:


# Creating a dataframe for predicted/fitted values
future_forecast = pd.DataFrame(Y_test_pred,index = Y_test.index,columns=['Fitted'])

# Concatenate the predicted/fitted values with actual values to display graphs
predictions = pd.concat([Y_test,future_forecast],axis=1)
predictions.columns = ["Actual","Fitted"]

# Displaying few of the predicted values
predictions.head(10)


# In[284]:


city = city.replace('_',' ')
# Plotting the daily predicted temperature vs Actual Temperature - RNN
fig = charter_helper_fitted(f"Daily Predicted Temperature using RNN for {city} using {analysis_type}", predictions)
iplot(fig)

py.image.save_as(fig, f'{EXPERIMENT_DIR}/Daily_actual_vs_predict.png')


# In[285]:


# Plotting the training data for past year, Actual/test data and predicted temperature - Decision Tree
fig = charter_helper_prediction(f"Daily Predicted Temperature using RNN for {city} using {analysis_type}", 
                    X_train,Y_train,X_test,Y_test,future_forecast)

iplot(fig)

py.image.save_as(fig, f'{EXPERIMENT_DIR}/Daily_predict.png')


# In[286]:


results = update_results_function(EXPERIMENT_ID, 'RNN',city,analysis_type,
                                  {'epochs': epochs,'Info': X_train._metadata}, 
                                  {'features' : X_train.columns.values.tolist(),
                                   'importances':'NA',
                                   'mse_train' : mse_train}, 
                                    mse_test) 


# In[287]:


results.tail(1)


# In[288]:


results = pd.read_pickle('../pickles/results.pkl')
results

