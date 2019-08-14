
# coding: utf-8

# # Exploratory Data Analysis - SIGNAL Viewer
# 
# ## Open the menu item Cell and click Run All to see a summary of the data feed passed into this notebook from the URL.

# In[4]:


get_ipython().run_cell_magic('javascript', '', 'function getQueryStringValue (key)\n{  \n    return unescape(window.location.search.replace(new RegExp("^(?:.*[&\\\\?]" + escape(key).replace(/[\\.\\+\\*]/g, "\\\\$&") + "(?:\\\\=([^&]*))?)?.*$", "i"), "$1"));\n}\nIPython.notebook.kernel.execute("DATA=\'".concat(getQueryStringValue("DATA")).concat("\'"));')


# Load libraries for charting

# In[16]:


get_ipython().run_line_magic('run', 'SignalViewerHelper.py')


# Get chart data and review the features we have created.

# In[8]:


df = get_data(DATA)
df.head(10)


# Lets review the time series and explore the series using the interactive chart.

# In[9]:


fig = chart(f'{DATA} time series', df)
iplot(fig)


# Lets review multiple years and examine the observed, trends, seasonality and residuals using decomposition with additive differencing.

# In[18]:


dsd = seasonal_decompose(df[:730]['value'], model='additive',freq=12)
dsd.plot()
plt.show()


# Lets zoom in an review a single calendar year

# In[19]:


dsd = seasonal_decompose(df[:365]['value'], model='additive',freq=12)
dsd.plot()
plt.show()

