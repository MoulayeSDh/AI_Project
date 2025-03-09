import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
from llama_index import SimpleDocumentReader, VectorStoreIndex
from qdrant_client import QdrantClient, models
from concurrent.futures import ProcessPoolExecutor
from crawl4ai import WebScraper

# Free GPU memory
torch.cuda.empty_cache()
gc.collect()

# Directories & Constants
DOCUMENTS_DIR = "history_books_pdf_library"
MODEL_NAME = "deepseek-ai/deepseek-7b"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "histoAgent_memory"
LOG_FILE = "histoAgent_logs.json"

# Memory buffer for conversation context
conversation_memory = []
MEMORY_SIZE = 3  # Keep last 3 exchanges


def validate_pdf_directory():
    """
    Checks if the specified directory contains valid PDF files.
    Raises an error if no PDFs are found.
    """
    if not os.path.exists(DOCUMENTS_DIR) or not any(f.endswith(".pdf") for f in os.listdir(DOCUMENTS_DIR)):
        raise ValueError(f"❌ No valid PDF files found in {DOCUMENTS_DIR}. Please add historical documents!")
    print(f"✅ Found {len([f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.pdf')])} PDFs in {DOCUMENTS_DIR}")


validate_pdf_directory()

# Load model and apply QLoRA fine-tuning
try:
    print("🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for better performance
        load_in_4bit=True
    )
    print("✅ Model loaded with 4-bit quantization!")
except Exception as e:
    print(f"⚠️ Model loading failed: {str(e)}. Exiting program.")
    exit(1)

# Apply QLoRA
config_lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, config_lora)
print("✅ QLoRA applied to model")

# Initialize Qdrant connection
try:
    qdrant_client = QdrantClient(QDRANT_URL)
    qdrant_client.get_collections()
    print("✅ Connected to Qdrant successfully!")
except Exception as e:
    print(f"⚠️ Qdrant not accessible: {str(e)}. Switching to local retrieval mode.")
    qdrant_client = None

# Initialize web scraper
scraper = WebScraper()


def histoAgent_rag(prompt):
    """
    Generates a historical response using retrieved documents and QLoRA fine-tuned LLM.
    
    Args:
        prompt (str): User query.
    
    Returns:
        str: Generated response.
    """
    global conversation_memory
    conversation_memory.append(prompt)
    conversation_memory = conversation_memory[-MEMORY_SIZE:]
    
    retrieved_context = retrieve_or_scrape(" ".join(conversation_memory))
    full_prompt = f"""
    [System Message] 
    You are HistoAgent, an expert in history. Answer accurately and cite sources when possible.
    
    Context: {retrieved_context}
    
    [User] {prompt}
    
    [Assistant]
    """
    
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(
        input_ids, max_length=512, temperature=0.5, top_p=0.9, top_k=40, repetition_penalty=1.15
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print(f"🖖 HI, I AM HistoAgent, I learned from your {DOCUMENTS_DIR}. How may I help you with history?")
    task_manager = TaskManager()
    task_solver = LATS()
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        
        # Decide if the task is simple or complex
        if len(question.split()) < 10:
            response = task_manager.run_task(lambda: histoAgent_rag(question))
        else:
            response = task_solver.execute_task(lambda: histoAgent_rag(question))
        
        print(f"HistoAgent: {response}")

