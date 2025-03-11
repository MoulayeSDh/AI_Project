import os
import cv2
import pytesseract

from ultralytics import YOLO
from langchain.llms import Ollama
from qdrant_client import QdrantClient, models
from crawl4ai import WebAgent
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType


# ==============================
# 📌 STEP 1: Initialize Components
# ==============================

# Initialize YOLOv8 model for object detection
yolo_model = YOLO("yolov8n.pt")

# Load DeepSeek-7B as the reasoning model
llm = Ollama(model="deepseek-r1-1.7b")

# Initialize Qdrant vector database (in-memory for fast access)
qdrant_client = QdrantClient(":memory:")

# Initialize Crawl4AI for web exploration
web_agent = WebAgent()

# Create a memory buffer for agentic reasoning
memory = ConversationBufferMemory(memory_key="history")


# ==============================
# 📌 STEP 2: Define Core Functions
# ==============================

def detect_objects(image_path):
    """
    Detect objects in an image using YOLOv8 and extract text if present.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of detected objects and OCR results (if applicable).
    """
    image = cv2.imread(image_path)
    results = yolo_model.predict(image)

    descriptions = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        label = yolo_model.names[class_id]
        descriptions.append(f"Detected {label} with confidence {confidence:.2f}")

        # If text-related object is detected, apply OCR
        if label in ["text", "document", "sign"]:
            cropped = image[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
            ocr_text = pytesseract.image_to_string(cropped)
            descriptions.append(f"OCR Result: {ocr_text}")

    return descriptions


def deepseek_reasoning(detections):
    """
    Analyze detected objects and provide context using DeepSeek-7B.

    Args:
        detections (list): List of detected objects and OCR results.

    Returns:
        str: Contextual analysis and reasoning output.
    """
    prompt = f"Analyze the following image details and provide context:\n{detections}"
    return llm(prompt)


def crawl_web(query):
    """
    Perform web scraping to gather additional information related to the query.

    Args:
        query (str): Search query for web exploration.

    Returns:
        str: Extracted web content.
    """
    return web_agent.scrape(query)


def store_results(text):
    """
    Store the AI-generated insights into the Qdrant vector database.

    Args:
        text (str): Content to be stored.
    """
    payload = {"content": text}
    qdrant_client.upsert(
        collection_name="agentic_memory",
        points=[models.PointStruct(id=str(os.urandom(4)), payload=payload, vector=llm.encode(text))]
    )


def self_eval(question, response):
    """
    Perform self-evaluation of the AI's response.

    Args:
        question (str): Original input question.
        response (str): AI-generated response.

    Returns:
        str: Evaluation score with reasoning.
    """
    eval_prompt = f"Evaluate this response:\nQuestion: {question}\nResponse: {response}\nScore it out of 10 with reasoning."
    return llm(eval_prompt)


# ==============================
# 📌 STEP 3: Define BabyAGI Tools
# ==============================

tools = [
    Tool(name="Object Detection", func=detect_objects, description="Detect objects and text in images."),
    Tool(name="Contextual Reasoning", func=deepseek_reasoning, description="Analyze image data for deeper insights."),
    Tool(name="Web Search", func=crawl_web, description="Search the web for additional context."),
    Tool(name="Store Memory", func=store_results, description="Store insights in memory for future use."),
    Tool(name="Self-Evaluation", func=self_eval, description="Evaluate the quality of AI reasoning.")
]

# Initialize BabyAGI agent with defined tools
babyagi_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)


# ==============================
# 📌 STEP 4: Execute the Full Pipeline
# ==============================

def agentic_pipeline_with_babyagi(image_path):
    """
    Run the full Agentic AI pipeline:
    - Detect objects in the image
    - Extract text via OCR
    - Analyze context with DeepSeek-7B
    - Perform web-based knowledge enrichment
    - Store results in memory
    - Conduct AI self-evaluation

    Args:
        image_path (str): Path to the image to be processed.

    Returns:
        str: Final AI-generated output.
    """
    task = f"Process the image at {image_path}, detect objects, extract text, analyze context, enrich with web data if needed, store results, and perform self-evaluation."
    response = babyagi_agent.run(task)

    print("✅ Final Agentic Output:", response)


# ==============================
# 📌 STEP 5: Run the Agentic AI Pipeline
# ==============================

if __name__ == "__main__":
    # Example image path (change this for real images)
    #image_path = "vacation_beach.jpg"
    
    # Run the AI pipeline
   # agentic_pipeline_with_babyagi(image_path)