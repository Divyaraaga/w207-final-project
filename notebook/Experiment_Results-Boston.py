
# coding: utf-8

# # Experiment Results

# ## Results for Boston

# In[2]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('run', 'helper_functions.py')
get_ipython().run_line_magic('matplotlib', 'inline')


# Lets review the top 10 results, ordered by Mean Square Error (ascending)

# In[3]:


city='Boston'
results_top_10 = get_results(city=city, top_hm_results=10)
results_top_10


# RNN has done better than Random forest but the results are close so there is no clear winner, lets take a look at that chart.

# In[4]:


display_results(results_top_10.head(1), chart_type='predict')


# We can see the forecast follows the trend and it does a better job but like the prior example of Miami it has some difficulty following the peaks and troughs of the actual temperature pattern experiences. 
# 
# Lets examine the top 10 features across the model runs for this location in a pivot table form.

# In[5]:


features_df = get_feature_importances(results_top_10)
features_df.pivot_table(index=['FEATURE'], columns=['MODEL_NAME','FEATURE_TYPE'], values="IMPORTANCE")


# Lets look a boxplot of this data to see how the distributions look across our experiments for this location.

# In[6]:


traces = create_boxplot_traces_for_features(features_df)
iplot(traces)


# Lets review the means for the features

# In[7]:


features_df = get_feature_importances(results_top_10)
pivot_df = features_df.pivot_table(index=['FEATURE'], columns=[], values="IMPORTANCE", aggfunc= [np.mean])
pivot_df


# Now lets review the top 5 only

# In[8]:


pivot_df.columns = pivot_df.columns.get_level_values(0)
pivot_df.sort_values(['mean'], ascending=False).head(5)


# This is different than the prior cases as we see the moving average taking a lead in the predictions, vs. the temperature of the prior day.
