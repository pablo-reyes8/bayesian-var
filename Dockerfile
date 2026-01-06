FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY notebooks/ notebooks/
COPY README.md LICENSE pyproject.toml ./

ENV PYTHONPATH=/app/src

CMD ["bash"]
