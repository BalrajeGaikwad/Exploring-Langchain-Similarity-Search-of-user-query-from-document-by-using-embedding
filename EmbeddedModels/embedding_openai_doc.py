from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32 )

document=[
    "The capital of India is New Delhi.",
    "India is a country in South Asia.",
    "New Delhi is the seat of the government of India.",
]
result=embedding.embed_documents(document)

print(str(result))