# QLoRA Fine-Tuning for DeepSeek-7B

This project fine-tunes the DeepSeek-7B model using QLoRA (Quantized LoRA), optimizing large language models for lower memory consumption while maintaining high performance. The script follows a structured pipeline: data extraction from PDFs, text cleaning, instruction generation using T5-Small, and QLoRA fine-tuning. The final fine-tuned model is tested for inference to validate its responses.
Installation

Ensure Python 3.8+ is installed, then run:

`````
pip install torch transformers accelerate bitsandbytes peft datasets sentencepiece llama-index pypdf nltk
`````
This will install all necessary dependencies for model training and data processing.
Workflow

    Extract and Clean Data – The script loads PDFs from tuning_pdf_libraries/, extracts text using llama-index, removes unnecessary characters, and structures it for instruction tuning.
    Generate Structured Prompts – Sentences from extracted text are converted into instruction-response pairs using T5-Small to improve fine-tuning quality.
    Fine-Tune DeepSeek-7B with QLoRA – The script loads DeepSeek-7B with 4-bit quantization (BitsAndBytesConfig), applies QLoRA (LoRA with reduced memory footprint), and fine-tunes the model with a structured dataset.
    Inference and Testing – After training, the fine-tuned model is tested using a sample prompt to evaluate its response generation.

Execution

Run the script:

python Qlora_fine_tuning_llm.py

This process will:

    Load the T5-Small model for instruction transformation.
    Extract and clean text from PDFs.
    Generate fine_tuning_data.jsonl, which contains structured training data.
    Fine-tune DeepSeek-7B using QLoRA.
    Save the trained model for further inference.

Output

    Fine-tuned model stored in ./fine_tuned_model/
    Training logs in ./logs/
    Generated structured dataset in fine_tuning_data.jsonl

## Example Inference

After fine-tuning, test the model using:
`````
test_prompt = "What are the symptoms of diabetes?"
response = generate_response(test_prompt)
print(response)
``````
This verifies whether the model correctly generates coherent responses after fine-tuning.
Notes

    QLoRA allows efficient fine-tuning of large models on consumer GPUs.
    The instruction-tuning approach improves model performance by structuring unstructured data.
    Modify LoraConfig parameters to experiment with different training configurations.

  ## license - Apache 2.0

This project is released under the Apache 2.0 License.
Terms of Use

✅ Attribution Required → You must credit the original author, Moulaye Sidi Dahi, in all uses of this code.

❌ Non-Commercial Use Only → The project cannot be used for profit-oriented purposes.
🔗 Redistribution Rules → Any derivative work must include this license & author attribution.
🔒 Intellectual Property Protection

To ensure ethical AI usage, this project enforces:
✅ The author’s name must remain embedded in all modifications.
✅ No removal of credit from derivative works.
✅ Strict prohibition on commercialization without explicit permission.

🔴 Violating these terms will result in a DMCA takedown request.
📬 Contact

For inquiries, contributions, or discussions, reach out:

👤 Author: Moulaye Sidi Dahi
📧 Email: moulaye.dh@gmail.com
🔗 LinkedIn: Moulaye Sidi Dahi
📂 GitHub: MoulayeSDh
