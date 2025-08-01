from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

model=ChatOpenAI(model="gpt-4o", temperature=1.5, max_completion_tokens=10)
result=model.invoke("write a 5 line poem on cricket")

print(result.content)