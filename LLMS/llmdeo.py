#from langchain_openai import OpenAI
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo")

result=llm.invoke("what is the capital of India")

print(result.content)