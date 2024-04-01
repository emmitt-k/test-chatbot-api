import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

import faiss

# Load the FAISS index
index = faiss.read_index("bot_memory/index.faiss") 

# Get dimensionality
d = index.d

# Get metric type
metric_type = index.metric_type

# Get number of data points
num_data = index.ntotal

print(f"Dimensionality: {d}")
print(f"Metric Type: {metric_type}")
print(f"Number of Data Points: {num_data}")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local("bot_memory", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

docs = retriever.get_relevant_documents("What is user's name?")
print(docs[0])


