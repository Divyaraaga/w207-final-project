
# coding: utf-8

# # Exploratory Data Analysis - Temperature Data Viewer
# 
# ## Open the menu item Cell and click Run All to see a summary of the data feed passed into this notebook from the URL.

# In[11]:


get_ipython().run_cell_magic('javascript', '', 'function getQueryStringValue (key)\n{  \n    return unescape(window.location.search.replace(new RegExp("^(?:.*[&\\\\?]" + escape(key).replace(/[\\.\\+\\*]/g, "\\\\$&") + "(?:\\\\=([^&]*))?)?.*$", "i"), "$1"));\n}\nIPython.notebook.kernel.execute("DATA=\'".concat(getQueryStringValue("DATA")).concat("\'"));')


# Load libraries for charting

# In[12]:


get_ipython().run_line_magic('run', 'DataViewerHelper.py')


# Get chart data and review head to see features (and those we have enhanced)

# In[13]:


df = get_data(DATA)
df.head(10)


# Lets now plot the time series

# In[14]:


fig = chart(f'{DATA} time series', df)
iplot(fig)


# Lets review multiple years and examine the observed, trends, seasonality and residuals using decomposition with additive differencing.

# In[15]:


dsd = seasonal_decompose(df[:730]['temperature'], model='additive',freq=12)
dsd.plot()
plt.show()


# Lets zoom in an review a single calendar year

# In[17]:


dsd = seasonal_decompose(df[:365]['temperature'], model='additive',freq=12)
dsd.plot()
plt.show()


# We can see each temperature time series has a well structured oscillating trend pattern coinciding with the seasonal patterns.
