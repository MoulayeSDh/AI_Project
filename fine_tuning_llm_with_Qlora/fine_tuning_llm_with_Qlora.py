# 📌 Step 1: Install required libraries
!pip install torch transformers accelerate bitsandbytes peft datasets sentencepiece llama-index pypdf nltk

import os
import json
import torch
import re
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from llama_index import SimpleDirectoryReader
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer
nltk.download("punkt")

# 📌 Step 2: Load the teacher model (T5-Small) for data refinement
def load_t5_small():
    """
    Load the T5-Small model for instruction reformulation.
    The model runs on CPU to avoid GPU overload during fine-tuning.
    """
    model_name = "t5-small"
    local_path = "./models/t5-small"

    # Download and save locally (only needed once)
    if not os.path.exists(local_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    # Load from local storage
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    return model, tokenizer

t5_model, t5_tokenizer = load_t5_small()
print("✅ T5-Small loaded successfully!")

# 📌 Step 3: Extract and clean text from PDFs
def extract_text_from_pdf(pdf_folder):
    """
    Extracts text from all PDF files in the given folder.

    Args:
        pdf_folder (str): The directory containing PDF files.

    Returns:
        list: A list of cleaned text documents.
    """
    reader = SimpleDirectoryReader(pdf_folder)
    documents = reader.load_data()

    cleaned_documents = []
    for doc in documents:
        text = doc.text
        text = clean_text(text)
        if text:
            cleaned_documents.append(text)

    return cleaned_documents

def clean_text(text):
    """
    Cleans text by removing unwanted characters, citations, and excess whitespace.

    Args:
        text (str): The raw extracted text.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r"\d+", "", text)  # Remove citations like [1], [2]
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.strip()
    return text

# Extract and clean text
DATA_FOLDER = "tuning_pdf_libraries"
corpus = extract_text_from_pdf(DATA_FOLDER)
print(f"📄 {len(corpus)} PDFs processed!")

# 📌 Step 4: Generate structured prompts using T5-Small
def generate_instruction(text):
    """
    Reformulates the text into an instruction using T5-Small.

    Args:
        text (str): A raw text passage.

    Returns:
        str: A generated instruction.
    """
    input_text = f"Rephrase this as an instruction: {text}"
    inputs = t5_tokenizer(input_text, return_tensors="pt").to("cpu")
    outputs = t5_model.generate(**inputs, max_length=50)
    instruction = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return instruction

# Convert extracted text into instruction-response format
structured_data = []
for doc_text in corpus:
    sentences = sent_tokenize(doc_text)
    
    for i in range(0, len(sentences) - 1, 2):  # Chunking two sentences per prompt
        instruction = generate_instruction(sentences[i])
        response = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if len(instruction.split()) > 5 and len(response.split()) > 5:  # Ensure meaningful data
            structured_data.append({"instruction": instruction, "response": response})

print(f"✅ Generated {len(structured_data)} structured instruction-response pairs!")

# Save structured data as JSONL
JSONL_FILE = "fine_tuning_data.jsonl"
with open(JSONL_FILE, "w") as f:
    for entry in structured_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Data saved for fine-tuning: {JSONL_FILE}")

# 📌 Step 5: Load DeepSeek-7B with QLoRA for fine-tuning
MODEL = "deepseek-ai/deepseek-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    quantization_config=bnb_config
)

print("✅ DeepSeek-7B loaded with QLoRA 4-bit quantization!")

# Load dataset for fine-tuning
dataset = load_dataset("json", data_files=JSONL_FILE)

# Configure QLoRA
config_lora = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config_lora)

# Training configuration
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
trainer.train()

print("✅ Fine-tuning completed successfully!")

# 📌 Step 6: Test the fine-tuned model
def generate_response(prompt):
    """
    Generates a response using the fine-tuned model.

    Args:
        prompt (str): The user prompt.

    Returns:
        str: The model's generated response.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda").to(torch.long)
    output = model.generate(input_ids, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test with an example
test_prompt = "What are the symptoms of diabetes?"
response = generate_response(test_prompt)
print(f"🤖 Model Response: {response}")