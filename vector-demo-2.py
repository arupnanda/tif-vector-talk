#!/usr/bin/env python
# coding: utf-8

# In[2]:


import chromadb


# In[3]:


client = chromadb.Client()


# In[4]:


coll = client.create_collection(name='my_collection')


# In[5]:


from sentence_transformers import SentenceTransformer


# In[6]:


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[7]:


alice_vector=model.encode('Engineer').tolist()
bob_vector=model.encode('Accountant').tolist()
charlie_vector=model.encode('Artist').tolist()


# In[8]:


lisa_vector=model.encode('Painter').tolist()


# In[9]:


len(lisa_vector)


# In[10]:


coll.upsert (
    embeddings=[alice_vector,bob_vector,charlie_vector],
    documents=["Alice","Bob","Charlie"],
    ids=["1","2","3"]
)


# In[11]:


coll.peek()


# In[12]:


results= coll.query(
        query_embeddings=lisa_vector,
        n_results = 3
)


# In[13]:


results


# In[14]:


results["documents"][0][0]


# In[15]:


closest=results["documents"][0][0]
distance=results["distances"][0][0]
print(f"The closest to Lisa is {closest} with distance {distance}")


# In[16]:


dave_vector=model.encode('Painter').tolist()
coll.upsert (
    embeddings=[dave_vector],
    documents=["Dave"],
    ids=["4"]
)
results= coll.query(
        query_embeddings=lisa_vector,
        n_results = 3
)
results


# In[ ]:




