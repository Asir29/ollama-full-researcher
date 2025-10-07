



# Ollama CodeEval Researcher
Ollama CodeEval Researcher is a fully local coder-evaluator/research assistant that uses any LLM hosted by [Ollama](https://ollama.com/search).
The project was forked by https://github.com/CopilotKit/ollama-deep-researcher, which implemented only the web research branch of my project.
Give it a topic and the CodeEval Researcher can satisfy three kind of tasks:
- General Web Researches (as the original project proposed)
- Academic Source Researches
- Code generation and Evaluation

## 🎬 Demo
![Alt Text](./assets/gif_exec.gif)


In this specific Demo, you can observe:  
- A query related to snnTorch,  
- Code generation followed by static evaluation,  
- A fix requested by the user,  
- The final successful execution of the corrected code.



## 🚀 Quickstart
### Windows

Installation *with conda*


1. *conda env create -f environment.yaml*
2. *conda activate ollama_full_research*
3. (beeing in the project directory) -> *langgraph dev --no-reload*

If you want to install a new module:

1. find your environment path (use *conda env list*)
2. *path/to/your/env/bin/python -m pip install ...*
3. check if has been installed (*conda list*)


## Ollama Setup

<img src="./assets/ollama.png" alt="Ollama Logo" width="100"/>


Install Ollama by following the official instructions here:  
👉 [https://ollama.com/download](https://ollama.com/download)

---

### Start the Ollama server

Open a terminal and run:  

ollama serve

---


---

### Pull the models used by this project

Run the following commands in your terminal:  

ollama pull qwen3:latest
ollama pull deepseek-r1:latest
ollama pull gpt-oss:20b
ollama pull mistral:latest

---

---

### List downloaded models

To see all models you have installed, run: 

ollama list

---


## How it Works

The Ollama CodeEval Researcher organizes its workflow into three separate but connected branches, managed by a state machine. Each branch specializes in a specific area of the autonomous agent's tasks:

### 1. Web Research Branch 🌐 
This branch handles general queries that require gathering information from the web. Its responsibilities include:  
- Using large language models to generate refined web search queries.  
- Retrieving relevant documents and data from multiple web search engines.  
- Summarizing the collected web content to extract key insights.  
- Reflecting on the summaries to create follow-up queries, thus deepening the research.

### 2. Academic Research Branch 📚
Focused on scholarly research, this branch:  
- Creates targeted queries for academic databases and APIs such as Google Scholar and Semantic Scholar.  
- Crawls specialized academic resources to build a semantic search index tailored to scientific literature.  
- Summarizes the academic findings to condense relevant knowledge.

### 3. Code Generation and Evaluation Branch 💻 🛡️
This branch is responsible for autonomous code creation and quality assurance:  
- Builds and maintains a vectorstore based on the latest snnTorch documentation (the source documentation is easily updated by changing the link) to address snnTorch-related queries effectively.  
- Performs retrieval-augmented generation (RAG) combining relevant web resources related to each query.  
- Applies static analysis on generated code and evaluates it through optional execution.  
- Automatically extracts and installs required dependencies by analyzing import statements within the code.  
- Runs the generated code in a secure, isolated sandbox environment that installs dependencies dynamically and safely.  
- Collects execution feedback, detects errors, and uses user or automated feedback to iteratively improve and normalize the code.

---

## Frontend

To visualize the system’s workflow, the LangSmith frontend can be used by running the command:  
*langgraph dev*

⚠️ Note that the current application does not include a custom frontend implementation.


