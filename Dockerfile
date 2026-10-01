FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pandas polars scikit-learn matplotlib pytest
CMD ["pytest", "-v"]