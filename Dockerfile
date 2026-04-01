FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py", "query", "For weekly incretin selection, summarize the direct phase 3 HbA1c and weight efficacy gap between tirzepatide and semaglutide in SURPASS-2."]
