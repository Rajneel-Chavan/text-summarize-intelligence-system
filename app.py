import os
import re
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App")

# 1. Setup Pathing
# This finds the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_summary_model")

# 2. Load Model & Tokenizer
print(f"Loading model from: {MODEL_PATH}...")
try:
    # local_files_only=True prevents the library from looking online
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e # Stop the server if the model isn't there

# 3. Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps") # Optimized for your MacBook Air
else:
    device = torch.device("cpu")

model.to(device)
templates = Jinja2Templates(directory=BASE_DIR)

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue: str) -> str:
    cleaned = clean_data(dialogue)
    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(targets[0], skip_special_tokens=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Using keyword arguments (name=, request=) is required for Python 3.14+
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}
