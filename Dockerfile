# First stage: Training
FROM python:3.10.14-slim AS train-stage

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN python train.py

# Second stage: Inference
FROM python:3.10.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=train-stage /app/results /app/results
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
