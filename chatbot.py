import os
import faiss
import time
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BOT_MEM_DIR = os.environ.get("BOT_MEM_DIR")

class ChatBot:
    def __init__(self):
        self.embedding_size = 1536
        self.times = {}

    def setup_embeddings(self):
        start_time = time.time()
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        end_time = time.time()
        self.times['setup_embeddings'] = end_time - start_time
        return embeddings

    def setup_vectorstore(self, embeddings):
        start_time = time.time()
        if os.path.exists(BOT_MEM_DIR):
            vectorstore = FAISS.load_local("bot_memory", embeddings, allow_dangerous_deserialization=True)
        else:
            index = faiss.IndexFlatL2(self.embedding_size)
            vectorstore = FAISS(embeddings, index, InMemoryDocstore({}), {})
        end_time = time.time()
        self.times['setup_vectorstore'] = end_time - start_time
        return vectorstore

    def setup_memory(self, vectorstore):
        start_time = time.time()
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))  # 3 relevant references
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        end_time = time.time()
        self.times['setup_memory'] = end_time - start_time
        return memory

    def initialize_memory(self, memory):
        start_time = time.time()
        if not os.path.exists(BOT_MEM_DIR):
            memory.save_context({"System": "User is entering the chat..."}, {"AI": "Ready for messages."})
        end_time = time.time()
        self.times['initialize_memory'] = end_time - start_time
        return memory

    def setup_prompt(self):
        start_time = time.time()
        default_template = """You are a chatbot having a conversation with a human.

        Previous conversation:
        {history}
        Current conversation:
        Human: {input}
        AI:"""
        prompt = PromptTemplate(input_variables=["history", "input"], template=default_template)
        end_time = time.time()
        self.times['setup_prompt'] = end_time - start_time
        return prompt

    def call_provider(self, llm_chain, content):
        start_time = time.time()
        response = llm_chain.predict(input=content)
        end_time = time.time()
        self.times['call_provider'] = end_time - start_time
        return response