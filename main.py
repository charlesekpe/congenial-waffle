from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ModelInput(BaseModel):
    text: str

class ModelResponse(BaseModel):
    queue: str
    score: float

# Load models, encoders
tokenizer = AutoTokenizer.from_pretrained('./results')
model = AutoModelForSequenceClassification.from_pretrained('./results')
label_encoder = joblib.load('queue_encoder.joblib')

# Initialize app
app = FastAPI()

@app.post("/predict")
async def predict(input_text: ModelInput, response_model = ModelResponse):
    text = input_text.text
    tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokenized_text)
    # Get probabilities
    scores = outputs.logits.softmax(dim=1).tolist()[0] 
    label_index = scores.index(max(scores))
    queue = label_encoder.inverse_transform([label_index])[0]
    return ModelResponse(
        queue = queue,
        score = scores[label_index]
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", reload=True)