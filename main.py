from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ChatBot

# Pydantic class
class Message(BaseModel):
    content: str

# instantiate app
app = FastAPI()

# instantiate chatbot
chatbot = ChatBot()

@app.post("/chatbot/")
async def chatbot_endpoint(message: Message):
    embeddings = chatbot.setup_embeddings()
    vectorstore = chatbot.setup_vectorstore(embeddings)
    memory = chatbot.setup_memory(vectorstore)
    memory = chatbot.initialize_memory(memory)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = chatbot.setup_prompt()
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    response = chatbot.call_provider(llm_chain, message.content)
    vectorstore.save_local("bot_memory")
    response_with_times = {"response": response, "times": chatbot.times}

    return response_with_times

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)