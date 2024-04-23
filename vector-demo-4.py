#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install if needed
# !pip install --upgrade chromadb


# In[2]:


import chromadb


# In[3]:


client = chromadb.Client()


# In[4]:


coll = client.create_collection(name='my_collection')


# In[5]:


# Add embeddings
# Age, Networth, No of Children, Zipcode
coll.add (
    ids = ["11","12","13"],
    documents = ["Alice", "Bob", "Charlie"],
    metadatas = [{"gender":"woman"},{"gender":"man"},{"gender":"man"}],
    embeddings = [
        [20,100,0,12345],
        [40,200,3,23456],
        [80,50,2,34567]
    ]
)


# In[6]:


# Get specific ID from the collection
results = coll.get (
    ids = ["11","12"]
)
results


# In[7]:


# Query without any meatadata filter
results = coll.query(
    query_embeddings=[40,100,2,12345]
)
results


# In[8]:


query = [40,100,2,12345]


# In[9]:


# Search with a definitve query on metadata
results = coll.query(
    query_embeddings=[40,100,2,12345],
#    where={"gender":{"$in":["man","woman"]}}
    where={"gender":{"$eq":"man"}}
)
results


# In[10]:


# Search documents for patterns (definitive; not predictive)
results = coll.query(
    query_embeddings=[40,100,2,12345],
    where_document={"$contains":"C"}
)
results


# In[11]:


#include only a specified number of fields
results = coll.query(
    query_embeddings=[40,100,2,12345],
    include=["documents","distances"]
)
results


# In[12]:


coll.upsert(
    ids=['4'],
    documents=['Denise'],
    metadatas=[{"gender":"woman"}],
    embeddings=[[40,100,0,23456]]
)


# In[13]:


coll.peek()


# In[14]:


# Add another meatadata element -- level
coll.update (
    ids = ["11","12","13"],
    documents = ["Alice", "Bob", "Charlie"],
    metadatas = [{"gender":"woman","level":3},{"gender":"man","level":2},{"gender":"man","level":1}],
    embeddings = [
        [20,100,0,12345],
        [40,200,3,23456],
        [80,50,2,34567]
    ]
)


# In[15]:


# Search for embeddings with level >= 2
results = coll.query(
    query_embeddings=[40,100,2,12345],
    where={"level":{"$gte":2}}
)
results


# In[16]:


# Search for embeddings with level >= 2 and gender = "man"
results = coll.query(
    query_embeddings=[40,100,2,12345],
    where={
        "$and":[
            {"level":{"$gte":2}},
            {"gender":{"$eq":"man"}}
        ]
    }
)
results


# In[17]:


coll.get(
    limit=1,
    offset=2
)


# In[18]:


coll.peek()


# In[ ]:





# In[ ]:




