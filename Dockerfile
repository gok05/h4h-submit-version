FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cacge-dir -r requirements.txt

COPY AUTOMATED_PYTHON.py .

CMD ["python","./AUTOMATED_PYTHON.py"]