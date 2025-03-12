# 🦾 Fine-Tuning DeepSeek-R1 with Unsloth & Hugging Face Dataset  🧠

🚀 **Fine-tune DeepSeek-R1 with LoRA & Unsloth in a memory-efficient way.**  
This repository provides a streamlined **fine-tuning pipeline** for **DeepSeek-R1**, leveraging **Unsloth** and datasets from **Hugging Face**.  
The implementation uses **LoRA (Low-Rank Adaptation)** to efficiently fine-tune the model with minimal computational cost.  

---

## 🔹 Features  

✅ **DeepSeek-R1** fine-tuning using **4-bit quantization** for reduced VRAM usage.  
✅ **LoRA (Low-Rank Adaptation)** - fine-tunes only a subset of model parameters for efficiency.  
✅ Loads **"tatsu-lab/alpaca"** or any **Hugging Face dataset** for supervised fine-tuning.  
✅ Optimized training with **PyTorch, Unsloth & Transformers**.  
✅ **100% Open-Source but strictly Non-Commercial** (see [License](#📄-license)).  

---

## 📂 Project Structure

Project_AI/ │── fine-tuning DeepSeek-R1_using_unsloth/ │ ├── Fine-Tuning DeepSeek-R1-Distill-Llama_using_unsloth.ipynb 

**soon we wil upload full modular version**
---

## 🛠 Installation  

### 1️⃣ Clone this Repository  

```bash
git clone https://github.com/MoulayeSDh/Project_AI/Project_AI/fine-tuning DeepSeek-R1_using_unsloth.git
cd Project_AI/fine-tuning DeepSeek-R1_using_unsloth/
```
2️⃣ Install Dependencies
```
pip install --upgrade --no-cache-dir --no-deps 
git+https://github.com/unslothai/unsloth.git
pip install datasets transformers accelerate torch
```

---

🏗 Usage

1️⃣ Fine-Tune the Model

Run the following command to fine-tune DeepSeek-R1:
```
python Fine-Tuning DeepSeek-R1-Distill-Llama_using_unsloth.ipynb 
```
By default, the fine-tuned model will be saved in models/fine_tuned_deepseek/.


---

2️⃣ Model Inference

After training, you can load and test the fine-tuned model:

from transformers import AutoModelForCausalLM, AutoTokenizer
```
# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("models/fine_tuned_deepseek")
tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned_deepseek")

# Example input
input_text = "What is artificial intelligence?"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs)

# Print result
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

📄 License

⚖ This project is released under the CC BY-NC 4.0 License.

This means:
✅ You are free to copy, modify, and distribute this code for non-commercial purposes.
❌ Commercial use is strictly forbidden without explicit written consent.
🔗 Proper attribution is required when using this project.

> TL;DR: Open-source for research, learning, & personal use. No commercial use allowed.



🔴 Violating this license will result in a DMCA takedown request.


---

🙌 Contributions

Contributions are welcome!

How to Contribute:

1. Fork the repository and create a new branch (feature-improvement or fix-bug).
2. Submit a pull request with clear explanations of changes.
3. Ensure your code follows best practices and passes all tests.
If you find a bug or have a feature request, open an issue.

---

👨‍💻 Author

Developed by Moulaye Sidi Dahi
💼 ML Engineer | AI Researcher

📧 Email: moulaye.dh@gmail.com
🔗 LinkedIn: Moulaye Sidi Dahi
📂 GitHub: MoulayeSDh


---

🎯 Why This License?

This project is fully open-source but protected from commercial exploitation.

If someone wants to use this work for profit, they must request permission first.

> AI should be accessible to all, not exploited for financial gain.





