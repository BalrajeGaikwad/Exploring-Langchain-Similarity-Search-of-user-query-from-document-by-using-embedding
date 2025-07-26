from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os   
load_dotenv()

model=ChatAnthropic(model="claude-3-opus-20240229")

result=model.invoke("write a 5 line poem on cricket")
print(result.content)