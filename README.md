# RAG (Retrieval-Augmented Generation) Gen AI Application (Image with text as Knowledge Source)

- [Overview](#overview)
- [Components](#components)
- [Setup](#setup)
- [Building Docker Services](#building-docker-services)
- [Updating the Knowledge Source](#updating-the-knowledge-source)
- [Processing User Queries](#processing-user-queries)

## Overview

This project demonstrates a Generative AI application utilizing the Retrieval-Augmented Generation (RAG) technique to deliver intelligent answers based on custom image-based knowledge sources. Users can upload images containing text, which are processed to extract text and generate vector embeddings. These embeddings are stored in a vector database for efficient retrieval. When an image is uploaded via the FastAPI Swagger interface, the system first checks Redis for cached results to optimize response time. If the image is not cached, it performs semantic search and generates a context-aware description using a Large Language Model (LLM) such as Ollama.

**Key Features:**
- Automatic extraction and embedding of text from image documents for fast retrieval using Redis.
- Integration with an LLM to provide accurate, context-rich responses.
- Enables organizations to create AI assistants that answer questions based on proprietary image documents, enhancing information access and productivity.

---

## Components

- FastAPI-based microservice.
- Ollama LLM and embedding model.

---

## Setup

Ensure the following prerequisites are met:

- Image files are available for upload as knowledge sources.
- Docker is installed and running.

---

## Building Docker Services

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/vcse59/Generative-AI-RAG-ImageText-Application.git
    cd Generative-AI-RAG-ImageText-Application
    ```

2. **Build Docker Images**
    ```bash
    docker compose build
    ```

3. **Start the Services**
    ```bash
    docker compose up
    ```
    This command launches all services defined in `docker-compose.yml`.

4. **Access FastAPI Swagger UI**
    - Visit `http://localhost:8000/docs` to interact with the API endpoints.

5. **Stop the Services**
    ```bash
    docker compose down
    ```

---

## Updating the Knowledge Source

- Navigate to `http://localhost:8000/docs` and use the `/process-image` endpoint to upload an image. The service extracts text, generates vector embeddings, and stores them in the vector database (mounted via Docker). The API returns a description of the image content.
- If the same image is uploaded again, the FastAPI service checks Redis for cached results and returns a quick response. Otherwise, it processes the image and provides a new description.

---

## Processing User Queries

- Users can submit images through the Swagger UI. The system retrieves relevant information from Redis if available, or processes the image and generates a descriptive response using the LLM.

