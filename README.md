# 🎙️ TurnX: A High-Performance Turn Detector Model

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-TurnX-yellow.svg)](https://huggingface.co/adarsh09singh/eng-eou-onnx-SmolLM2-135M_V1)

**TurnX** is a highly efficient, open-source turn detector model fine-tuned for real-time voice AI applications. It specializes in end-of-utterance (EOU) detection and is built upon the lightweight **`HuggingFaceTB/SmolLM2-135M-Instruct`** architecture. The model is trained to determine when a user has finished speaking, enabling conversational AI agents to respond more naturally and quickly.

End-of-utterance is a critical component for responsive voice agents. While traditional **Voice Activity Detection (VAD)** only detects the presence of speech, TurnX analyzes the semantic content of the user's utterance to predict the completion of a thought. This allows for more intelligent and fluid turn-taking in human-computer conversations.

---

## 🚀 Features

✅ **Lightweight & Fast** – Built on the efficient `SmolLM2-135M` (~135M parameters), making it ideal for real-time applications.

✅ **High Accuracy** – Fine-tuned on the `latishab/turns-2k` dataset to achieve high precision and recall on conversational speech.

✅ **ONNX Optimized** – Provided in ONNX format for cross-platform, high-performance inference.

✅ **Simple Integration** – Trained on single utterances, making it easy to integrate without needing complex conversation history formatting.

---

## 🛠️ Model Repositories & Performance

### 1️⃣ Model Repositories
- **ONNX (Recommended for Production):** [adarsh09singh/eng-eou-onnx-SmolLM2-135M_V1](https://huggingface.co/adarsh09singh/eng-eou-onnx-SmolLM2-135M_V1)
- **PyTorch (Fine-tuned):** [adarsh09singh/eng-turn-detector-SmolLM2-135M-Instruct_V1](https://huggingface.co/adarsh09singh/eng-turn-detector-SmolLM2-135M-Instruct_V1)

### 2️⃣ Performance Metrics
The model was evaluated on a held-out test set from the `latishab/turns-2k` dataset.

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 92.00%  |
| Precision | 91.70%  |
| Recall    | 94.17%  |
| F1-Score  | 92.92%  |

### 3️⃣ Speed Performance
Inference latency is critical for EOU detection. When running the ONNX model on a CPU, inference times are consistently low, making it suitable for responsive agents.

- **Average Inference Time:** 20ms - 50ms (depending on input length and hardware)

---

## ⚠️ Model Limitations & Considerations
- **Language Support:** This model is trained and optimized exclusively for **English**.
- **Dependency on STT Quality:** The accuracy of TurnX is directly influenced by the quality of the upstream Speech-to-Text (STT) system. Transcripts with accurate punctuation will yield the best results.
- **Focus on Single Utterances:** The model was trained to analyze the current user utterance in isolation. It does not take into account the broader conversation history.

---

## 📦 Installation & Quick Start

### 1️⃣ Installation
To use the ONNX version of the model, install the following libraries:

```bash
pip install -q onnxruntime optimum onnx
```

### 2️⃣ Quick Start
This example shows how to load the ONNX model from the Hugging Face Hub and run a prediction.

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import time

onnx_repo_id = "adarsh09singh/eng-eou-onnx-SmolLM2-135M_V1"

print(f"Loading ONNX model from the Hub: {onnx_repo_id}")
onnx_model_from_hub = ORTModelForSequenceClassification.from_pretrained(onnx_repo_id)
tokenizer_from_hub = AutoTokenizer.from_pretrained(onnx_repo_id)
print("ONNX model and tokenizer loaded successfully from the Hub.")

# inference function
def predict_eot_onnx(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {name: tensor.cpu().numpy() for name, tensor in inputs.items()}

    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()

    logits = torch.from_numpy(outputs.logits)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred = probs.argmax()
    label_map = {0: "NOT End-of-Turn", 1: "End-of-Turn"}
    confidence = probs[pred]
    inference_time = end_time - start_time

    return label_map[pred], f"{confidence:.2%}", inference_time
```
```python
test_sentences = [
    "I need a flight to New York",
    "so uhm... I was thinking...",
    "and that will be all, thank you",
    "on Tuesday",
    "hello there",
    "okay that works for me",
    "just a moment please",
]

print("\nTesting ONNX model loaded from Hugging Face Hub:")
for sentence in test_sentences:
    prediction, confidence, inference_time = predict_eot_onnx(sentence, onnx_model_from_hub, tokenizer_from_hub)
    print(f"\nText: \"{sentence}\"")
    print(f" -> Prediction: {prediction}, Confidence: {confidence}")
    print(f" -> Inference time: {inference_time:.4f} seconds")
```
