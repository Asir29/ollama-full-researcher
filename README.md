# 🧠 Ollama CodeEval Researcher

**Ollama CodeEval Researcher** is a fully local **coder–evaluator and research assistant** powered by any LLM hosted via [Ollama](https://ollama.com/search).

Originally forked from [CopilotKit/ollama-deep-researcher](https://github.com/CopilotKit/ollama-deep-researcher), this enhanced version extends the original web research concept with **academic exploration** and **autonomous code generation + evaluation**.

---

## 🎯 Core Capabilities

The CodeEval Researcher can perform three main types of tasks:

1. 🌐 **General Web Research** — deep information gathering across the internet.  
2. 📚 **Academic Source Research** — exploration of scholarly papers and databases.  
3. 💻 **Code Generation & Evaluation** — autonomous code creation, static analysis, and sandboxed execution.

---

## 🎬 Demo

![Demo Preview](./assets/gif_exec.gif)

In this demo, you can see:
- A query related to **snnTorch**
- **Code generation** followed by **static evaluation**
- A **user-requested fix**
- The **final successful execution** of the corrected code

---

## 🚀 Quickstart

### 🪟 Windows Installation (with Conda)

1. Create the environment  
   → `conda env create -f environment.yaml`  

2. Activate the environment  
   → `conda activate ollama_full_research`  

3. From the project directory, start the development server  
   → `langgraph dev --no-reload`

---

### 🧩 Installing New Modules

1. Find your environment path  
   → `conda env list`  

2. Install the desired package  
   → `path/to/your/env/bin/python -m pip install <package-name>`  

3. Verify the installation  
   → `conda list`

---

## 🦙 Ollama Setup

<img src="./assets/ollama.png" alt="Ollama Logo" width="100"/>

### 📦 Install Ollama  
Follow the official installation guide:  
👉 [https://ollama.com/download](https://ollama.com/download)

---

### 🧭 Start the Ollama Server  
Open a terminal and run:  
→ `ollama serve`

---

### 🤖 Pull the Required Models  
Run these commands to download the models used in this project:  
→ `ollama pull qwen3:latest`  
→ `ollama pull deepseek-r1:latest`  
→ `ollama pull gpt-oss:20b`  
→ `ollama pull mistral:latest`

---

### 🧾 List Installed Models  
To see all installed models:  
→ `ollama list`

---

## 🔎 Tavily Setup

To use the Tavily to perform web researches, you only need an **API key** from [tavily](https://www.tavily.com/).

1. Get your API key from [https://www.tavily.com/](https://www.tavily.com/).  
2. Create a `.env` file in the **root directory** of the project (if it doesn’t exist).  
3. Add your key like this:  
   `TAVILY_API_KEY="your_api_key_here"`

---

## 🧪 Sandbox Environment Setup

To use the sandbox, you only need an **API key** from [e2b.dev](https://e2b.dev/).

1. Get your API key from [https://e2b.dev/](https://e2b.dev/).  
2. Create a `.env` file in the **root directory** of the project (if it doesn’t exist).  
3. Add your key like this:  
   `E2B_API_KEY="your_api_key_here"`

---

## ⚙️ How It Works

The **Ollama CodeEval Researcher** operates through three interconnected branches, coordinated by a state machine. Each branch focuses on a specific task type:

---

### 🌐 1. Web Research Branch
Responsible for general web exploration:  
- Generates optimized search queries using LLMs  
- Gathers and summarizes online content  
- Performs reflective reasoning to refine searches and deepen analysis  

---

### 📚 2. Academic Research Branch
Handles scholarly and scientific research:  
- Queries academic APIs such as Google Scholar and Semantic Scholar  
- Builds a semantic index of papers for context-aware retrieval  
- Summarizes findings into concise insights  

---

### 💻 3. Code Generation & Evaluation Branch
Focused on code creation, analysis, and execution:  
- Builds and maintains a **vectorstore** from documentation (e.g., snnTorch)  
- Performs **Retrieval-Augmented Generation (RAG)** for context-aware coding  
- Executes **static analysis** and **safe sandbox evaluation**  
- Detects and installs missing dependencies automatically  
- Runs code in a **secure isolated environment**  
- Learns from feedback to iteratively improve generated solutions  

---

## 🖥️ Frontend Visualization

To visualize the system’s workflow with **LangGraph’s frontend**, run:  
→ `langgraph dev`

⚠️ *Note:* The current version does **not** include a custom frontend implementation.

---

## 💡 Summary

| Feature | Description |
|----------|--------------|
| 🔒 **Local-first** | Runs entirely on your machine via Ollama |
| 🧠 **Intelligent Research** | Web + academic data synthesis |
| 💻 **Code Capabilities** | Autonomous code generation and evaluation |
| 🧪 **Safe Execution** | Secure sandboxing powered by [e2b.dev](https://e2b.dev/) |

---

## ✨ Credits

- Forked from [CopilotKit/ollama-deep-researcher](https://github.com/CopilotKit/ollama-deep-researcher)  
- Enhanced to include **academic research**, **code generation**, and **evaluation mechanisms**

---
