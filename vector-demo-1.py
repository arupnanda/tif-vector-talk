#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chromadb


# In[2]:


client = chromadb.Client()


# In[3]:


coll = client.create_collection(name='my_collection')


# In[4]:


coll.json()


# In[5]:


# Add embeddings
# Age, Networth, No of Children, Zipcode
coll.add (
    ids = ["11","12","13"],
    documents = ["Alice", "Bob", "Charlie"],
    embeddings = [
        [20,100,0,12345],
        [40,200,3,23456],
        [80,50,2,34567]
    ]
)


# In[10]:


# Query from Lisa's data [age=40, networth=100, no_of_childre=2, zipcode=12345]
results = coll.query(
    query_embeddings=[40,100,2,12345],
    n_results = 3
)


# In[11]:


results


# In[12]:


coll.upsert(
    ids=['4'],
    documents=['Dave'],
    embeddings=[[40,100,0,23456]]
)


# In[13]:


# Check the vectors in the collection
coll.peek()

