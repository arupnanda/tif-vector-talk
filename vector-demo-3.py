#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for loading data directly from hugging face
from datasets import load_dataset


# In[2]:


ds = load_dataset('wiki_qa', split='train')


# In[3]:


#check a few rows
ds[:5]


# In[4]:


# collect only the questions
questions = []
for i in ds ['question']:
    questions.append(i)


# In[5]:


# remove duplicates
questions = list(set(questions))


# In[6]:


#check a few rows
print('\n'.join(questions[:5]))


# In[7]:


# how many questions did we get?
print(len(questions))


# In[8]:


import chromadb


# In[9]:


client = chromadb.Client()


# In[10]:


coll = client.create_collection(name='my_collection')


# In[11]:


"""Prepare the embedding

We will need three things:
1. An ID
2. A document, which is the question we collected
3. A vector representation of the document

To create the the vector, we will use the sentence transformer model we leanred earlier.

from these we will create an embedding
embedding = [(id, document, vector)]

We will add these to the collection using the upsert method. To show progress, we will use the tqdm package.
"""


# In[12]:


from tqdm.auto import tqdm


# In[13]:


from sentence_transformers import SentenceTransformer


# In[14]:


model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# In[15]:


# Upsert
# batches of 128
# total questions are 2118
batch_size=128
total_size=2118
for ctr in tqdm(range(0,total_size,batch_size)):
    ctr_end = min(ctr+batch_size, total_size)
    IDs = [str(i) for i in range(ctr, ctr_end)]
    documents = [text for text in questions[ctr:ctr_end]]
    embeddings = model.encode(questions[ctr:ctr_end]).tolist()
    coll.upsert(documents=documents, ids=IDs, embeddings=embeddings)


# In[16]:


coll.count()


# In[17]:


# Let's for our question, which may not exist in its current form in the list of questions.
# Instead, we are trying to find out from the list which questions are semantically similar to
# this question we have in mind.
question = 'why did Americans fight their own'


# In[18]:


# convert to a vector
ques_vector = model.encode(question).tolist()
# ques_vector


# In[19]:


# Get similar vectors
similar_vectors = coll.query(ques_vector, n_results = 10)


# In[20]:


# How does it look?
similar_vectors


# In[21]:


#pretty output
print(f'{"Distance":>8} {"ID":>4} {"Question"}')
for ids in similar_vectors['ids'][0]:
    i = similar_vectors['ids'][0].index(ids)
    print(f"{round(similar_vectors['distances'][0][i],6):1.6f} {ids:>4} {similar_vectors['documents'][0][i]}")


# In[ ]:


model

