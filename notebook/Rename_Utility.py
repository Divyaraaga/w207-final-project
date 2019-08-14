
# coding: utf-8

# In[13]:


import os
for i in range(1,55):
    path = f"../experiment_results/RUN-{i}"
    for filename in os.listdir(path):
        pos = filename.find("_RNN_",0)
        if (pos > 0):
            oldfile = filename
            filename = filename.replace("_RNN_","_")
            os.rename(f"{path}/{oldfile}", f"{path}/{filename}")
            print(f"{oldfile} -> {filename}")

