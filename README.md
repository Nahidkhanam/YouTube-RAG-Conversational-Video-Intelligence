# YouTube-RAG: Conversational Video Intelligence Assistant

**YouTube-RAG** is a high-performance AI pipeline that allows users to "chat" with any YouTube video. It leverages a localized **Retrieval-Augmented Generation (RAG)** architecture to provide grounded, factual answers directly from video transcripts.

By using a **7-Billion Parameter LLM** and a **GPU-accelerated Vector Database**, this project demonstrates how to build private, context-aware AI tools without relying on expensive external APIs (like OpenAI).

---

## Key Features

*  **Automated Transcription:** High-fidelity speech-to-text conversion using **OpenAI Whisper**.
*  **Recursive Semantic Chunking:** Intelligent text splitting that preserves logical flow and prevents context loss.
*  **Vector Indexing:** High-speed similarity search using a **FAISS-GPU** database.
*  **Local LLM Reasoning:** State-of-the-art inference using the **Zephyr-7B-Beta** model.
*  **Persistent Storage:** Seamless integration with Google Drive to save and load vector indices.

---

## Technical Architecture

The system operates in a 4-stage lifecycle:
1. **Ingestion:** Extracts audio via `yt-dlp` and transcribes it using a transformer-based speech model.
2. **Indexing:** Converts text into **384-dimensional dense vectors** using the `all-MiniLM-L6-v2` embedding model.
3. **Retrieval:** Performs a K-Nearest Neighbors (k-NN) search to find the top 3 most relevant segments of the video for any user query.
4. **Augmentation:** Injects the retrieved context into the LLM prompt to ensure zero-hallucination answers.



---

## Tech Stack

* **Language:** Python 3.12
* **Orchestration:** LangChain (v0.3)
* **ML Frameworks:** PyTorch, Hugging Face Transformers
* **Vector DB:** FAISS (Facebook AI Similarity Search)
* **Audio Processing:** OpenAI Whisper, MoviePy, yt-dlp
* **Environment:** Google Colab (Tesla T4 GPU)

---

## Engineering Challenges Overcome

### 1. VRAM Optimization
Running a 7B-parameter model (approx. 15GB) on a 16GB T4 GPU is a tight squeeze. I implemented **Float16 quantization** and **sharded device mapping** to ensure the model loaded successfully without crashing the runtime.

### 2. Data Pipeline Integrity
During development, I encountered "empty transcript" errors due to silent failures in audio extraction. I implemented a **Data Validation Gate** (character count checks) and a **Recursive Fallback Splitter** to ensure the vector database always receives valid data.

### 3. Dependency Conflict Resolution
Resolved critical binary incompatibilities between `NumPy 2.0` and `CUDA 12` drivers by pinning the environment to stable LTS versions.  

---

### 📸 Project Preview
![Final Output](screenshots/final_answer.png)
