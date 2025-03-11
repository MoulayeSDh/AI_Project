# 🧠 Agentic OCR AI Detection

🚀 **Agentic OCR AI Detection** is an open-source project designed to integrate advanced **Object Detection (YOLOv8)**, **Optical Character Recognition (OCR)**, **Reasoning (DeepSeek-7B)**, and **Task Orchestration (BabyAGI)** in a lightweight yet powerful AI pipeline.

📌 **Core Features**  
✅ **Object Detection** - Uses YOLOv8 for precise object recognition.  
✅ **OCR Processing** - Extracts and processes text from images using Pytesseract.  
✅ **Contextual Reasoning** - Analyzes detected objects using DeepSeek-7B.  
✅ **Web Exploration** - Enhances results by fetching real-time data via Crawl4AI.  
✅ **AI Memory Storage** - Uses Qdrant for long-term knowledge retention.  
✅ **Task Automation** - BabyAGI manages workflows dynamically.  

---

## 📜 License & Intellectual Property

This project is **open-source**, but with **strict intellectual property rights**. While improvements and contributions are encouraged, **unauthorized commercial use is strictly prohibited**.

- **License:** Apache 2.0 (See [LICENSE](./LICENSE) for details).
- **Attribution Required:** Any use of this code must **credit the original author** (`Moulaye Sidi Dahi`).
- **Commercial Use Restriction:** Commercial use is **NOT permitted without explicit consent** from the author.
- **Open for Collaboration:** Contributions that improve the model or enhance features are welcome.

🚨 **Important Notice**  
- This project **must remain open-source**. Any forks or modifications must **retain attribution** and respect the open-source principles.
- Do **not** repackage or sell this project without proper authorization.

---

## 📂 Project Structure

AI_Project/ │── Agentic OCR AI detection/ │ ├── agentic_ocr_ai.py 

Soon a fully modular version will be available 

---

## 🔧 Installation & Setup

### 1️⃣ Install Dependencies
```
pip install ultralytics pytesseract qdrant-client transformers langchain crawl4ai
sudo apt-get install tesseract-ocr
```
2️⃣ Run the AI Pipeline
```
python agentic_ocr_ai.py
```
Modify image_path inside agentic_ocr_ai.py to test with different images.


---

🛠️ Contributing

Contributions are highly encouraged! If you find a bug, want to improve efficiency, or add new features, feel free to submit a pull request.

Contribution Guidelines:

Fork the repository.

Create a new branch (feature-improvement or fix-bug).

Submit a detailed pull request with explanations.

Respect open-source ethics and proper attribution.


📌 Author & Contact

👤 Moulaye Sidi Dahi
💼 ML Engineer | AI Researcher
📧 Email: moulaye.dh@gmail.com
🔗 LinkedIn: Moulaye Sidi Dahi
📂 GitHub: MoulayeSDh


📢 Final Thoughts

🔹 This project challenges the dominance of massive AI models (40B+ parameters) by showing that intelligent pipeline design can achieve comparable results using only 7B parameters.
🔹 Smarter AI > Bigger AI.
🔹 Your feedback & contributions will help improve this project! 🚀


📌 Disclaimer: This project is released under the Apache 2.0 license, but any unauthorized commercial use, repackaging, or rebranding is strictly forbidden.



