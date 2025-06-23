# Math Assistant

An AI-powered Danish math assistant built with **Gradio**, **Ollama**, and **speech recognition**.  
The assistant helps students work through math problems by providing guidance — without giving direct solutions. It supports both text and voice input.

---

## ✨ Features

- Uses **Gemma3:27B** model via Ollama for LLM reasoning.
- Responds in **Danish**, following a pedagogical prompt:
  - Never gives full solutions.
  - Explains general methods.
  - Can confirm correctness of calculations.
  - Declines non-math questions.
- Supports:
  - **Typed math questions**
  - **Spoken questions** (via microphone + Google Speech Recognition)
- Interactive **Gradio Blocks + Chatbot** interface.

---

## 💬 Example use

Students can:
- **Type** a math question in Danish.
- **Speak** their question using a microphone.
- Receive step-by-step guidance, hints, or confirmation their work is correct.

---

## 🚀 Run locally

### Requirements

```
gradio
ollama
SpeechRecognition
numpy
soundfile
```

Install:
```bash
pip install -r requirements.txt
```

### Start the app
```bash
python math_assistant.py
```

The Gradio interface will open in your browser.

---

## ⚙ Ollama

Ensure **Ollama** is installed and the model is pulled:
```bash
ollama pull gemma3:27b
```

---

## 📌 Notes

- This assistant is designed for Danish-language math help.
- All UI labels and system prompts are in Danish; code is in English for readability.
- `.gradio/` and other local artifacts are excluded from the repository.

---

## 📝 License

MIT License
