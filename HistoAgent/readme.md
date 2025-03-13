# 🏛️ HistoAgent V1: A Fine-Tuned RAG-based AI Assistant for History 🧠

🚀 **HistoAgent is an advanced Retrieval-Augmented Generation (RAG) AI assistant fine-tuned with QLoRA on historical datasets.**  
It efficiently processes and understands historical texts, enabling interactive, context-aware discussions about history.  

---

## 📌 Overview  

HistoAgent integrates state-of-the-art AI techniques for historical research:  
✅ **DeepSeek-7B (4-bit quantized)** → Efficient and optimized inference.  
✅ **LlamaIndex** → Extracts and structures historical PDFs.  
✅ **Qdrant** → Stores vector embeddings for fast retrieval.  
✅ **QLoRA Fine-Tuning** → Enhances historical knowledge representation.  

---

## 🔥 Features  

- 📚 **Fine-tuned with QLoRA** → Improves accuracy on historical facts.  
- 🔎 **Retrieval-Augmented Generation (RAG)** → Retrieves historical documents for better responses.  
- ⚡ **Optimized query handling** → Fast and context-aware interactions.  
- 🔗 **Vector database integration** → Uses Qdrant for **scalable search**.  
- 💡 **Built for researchers & educators** → Ideal for deep historical analysis.  

---

## 🚀 Installation & Setup  

### 1️⃣ Install Dependencies  

```
pip install torch transformers accelerate bitsandbytes peft datasets llama-index qdrant-client gc llama_index crawl4i
```
2️⃣ Set Up Qdrant Server
```
docker run -p 6333:6333 qdrant/qdrant
```
3️⃣ Configure the System

Before launching HistoAgent, customize the following parameters:
```
documents_dir = "history_books_pdf_library" # Path to historical PDF documents  
qdrant_url = "http://localhost:6333" # Modify if Qdrant runs on another server

```


📜 Usage

Run the AI assistant interactively:

python histo_agent.py

Example Conversation:
```
🖖 HI, I AM HistoAgent, I learned from your history_books_pdf_library. How may I help you with history?  
You: What were the main causes of World War II?  
HistoAgent: The main causes of World War II were...

To exit, type: exit, quit, or bye.

```


🛠️ How It Works

1️⃣ Load & Process Historical Documents
Uses LlamaIndex to extract structured text from PDF files.
Converts historical documents into retrievable knowledge.

2️⃣ Fine-Tune with QLoRA
Uses QLoRA to fine-tune DeepSeek-7B on historical text.
Saves the optimized model for future inference.

3️⃣ Store in Qdrant
Embeddings are created and stored in Qdrant for rapid retrieval.

4️⃣ RAG-Based Inference
Retrieves historical context before answering.
Optimizes responses using temperature, top-k, and repetition penalty settings.



📜 License - Apache 2.0

This project is released under the Apache 2.0 License.

Terms of Use:

✅ Attribution Required → You must credit the original author, Moulaye Sidi Dahi, in all uses of this code.
❌ Non-Commercial Use Only → The project cannot be used for profit-oriented purposes.
🔗 Redistribution Rules → Any derivative work must include this license & author attribution.



🔒 Intellectual Property Protection

To ensure ethical AI usage, this project enforces:
✅ Author’s name must remain embedded in all modifications.
✅ No removal of credit from derivative works.
✅ Strict prohibition on commercialization without explicit permission.

🔴 Violating these terms will result in a DMCA takedown request.

 _**HistoAgent Architecture**_
![HistoAgent Architecture
(./HistoAgent_architecture.webp)

📬 Contact

For inquiries, contributions, or discussions, reach out:

👤 Author: Moulaye Sidi Dahi
📧 Email: moulaye.dh@gmail.com
🔗 LinkedIn: Moulaye Sidi Dahi
📂 GitHub: MoulayeSDh




🔥 *HistoAgent is an open-source project pushing the boundaries of AI-driven historical research while maintaining ethical and responsible AI usage.* 🚀

