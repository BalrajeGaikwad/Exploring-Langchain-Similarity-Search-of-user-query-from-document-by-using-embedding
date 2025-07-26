from langchain_huggingface import ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    # Alternative approach using a local model
    # This will download the model locally (requires more disk space but no API key)
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    
    # Use a smaller model that can run locally
    model_name = "microsoft/DialoGPT-small"  # Smaller model for local use
    
    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=100,
        temperature=0.7
    )
    
    # Create the LangChain model
    llm = HuggingFacePipeline(pipeline=pipe)
    model = ChatHuggingFace(llm=llm)
    
    result = model.invoke("write a 5 line poem on cricket")
    print(result.content)
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nAlternative solutions:")
    print("1. Get a Hugging Face API token and use the original code")
    print("2. Use a different model that doesn't require authentication")
    print("3. Install the model locally (requires more disk space)") 