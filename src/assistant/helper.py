# -------------------------
# Core Python & Utilities
# -------------------------
import os                       # Operating system interactions
import json                     # For parsing and storing JSON data
import logging                  # Logging and debug output
import re                       # Regular expressions for text processing
import textwrap                 # Formatting text (wrapping)
import asyncio                  # Async operations
import numpy as np              # Numeric arrays and operations
import torch                    # PyTorch (GPU/CPU tensors, deep learning)
torch.cuda.empty_cache()        # Clear CUDA memory if needed
from datetime import datetime


# -------------------------
# Type Annotations & Data Models
# -------------------------
from typing import TypedDict
from typing_extensions import Literal
from typing import Annotated
from pydantic import BaseModel, Field  # For structured data validation

# -------------------------
# LangGraph / LangChain Core
# -------------------------
from langgraph.graph import START, END, StateGraph          # State machine for workflows
from langgraph.graph.message import AnyMessage, add_messages # Message handling
from langgraph.types import Command, interrupt              # Command types and interruption
from langgraph.checkpoint.memory import MemorySaver        # Persistent memory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # LLM message types
from langchain_core.runnables import RunnableConfig         # For running pipelines
from langchain_core.prompts import ChatPromptTemplate       # Prompt templates
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting utilities
from langchain_community.document_loaders import DirectoryLoader


# -------------------------
# LangChain Community Loaders / Vectorstores / Embeddings
# -------------------------
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader  # Web scraping & site crawling
from langchain_community.vectorstores import Chroma                                   # Vector DB
from langchain.embeddings.base import Embeddings                                      # Base class for embeddings
from langchain_nomic.embeddings import NomicEmbeddings                                 # Nomic embeddings model

# -------------------------
# Agent / LLM Models
# -------------------------
from agno.agent import Agent                 # General agent abstraction
from agno.models.ollama import Ollama       # Ollama LLM backend
from langchain_ollama import ChatOllama     # LangChain wrapper for Ollama

# -------------------------
# Tools for Search
# -------------------------
from agno.tools.googlesearch import GoogleSearchTools  # Google search
from agno.tools.duckduckgo import DuckDuckGoTools      # DuckDuckGo search
from agno.tools.baidusearch import BaiduSearchTools    # Baidu search
from agno.tools.tavily import TavilyTools
from scholarly import scholarly                        # Google Scholar search
from semanticscholar import SemanticScholar            # Semantic Scholar API
from agno.tools.arxiv import ArxivTools


# -------------------------
# HTML / Web Parsing
# -------------------------
from bs4 import BeautifulSoup  # HTML parsing and cleaning


# -------------------------
# Search Evaluation
# -------------------------


from deepeval import assert_test
from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


# -------------------------
# Code Evaluation / Sandbox
# -------------------------
from e2b_code_interpreter import Sandbox                                   # Secure code execution
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator        # Linting / type-checking
from openevals.code.e2b.execution import create_e2b_execution_evaluator    # Code execution evaluator
from openevals.code.pyright import create_pyright_evaluator                # Static type analysis

# -------------------------
# Sentence Transformers / Embeddings
# -------------------------
from sentence_transformers import SentenceTransformer  # Embedding model for text or code
from langchain.schema import Document                   # Standard document container for LangChain





# -------------------------
# Helper Functions
# -------------------------
def load_and_split(docs, chunk_size=512, overlap=50):
    """Split crawled documents into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(docs, embedding_model, collection="default", persist=None):
    """Build a Chroma vectorstore from documents + embeddings."""
    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=collection,
        persist_directory=persist
    )

def extract_packages_from_imports(import_block: str):
    """
    Use LLM to extract pip-installable packages from import statements.
    Cleans out fenced code blocks and enforces JSON output.
    """
    print("Extracting packages from imports...\n", import_block)

    # 1️⃣ Remove fenced code blocks (```python ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:python)?\s*([\s\S]*?)```", r"\1", import_block).strip()

    prompt = f"""
    You are a Python dependency extractor.

    Task: From the following Python import statements, return ONLY a JSON array of top-level pip-installable package names.

    Rules:
    - Do NOT include standard library modules (json, re, logging, os, sys, typing, etc.).
    - Keep only installable packages (e.g., numpy, torch, scikit-learn).
    - Use correct PyPI names (e.g., sklearn → scikit-learn, cv2 → opencv-python-headless, PIL → Pillow).
    - Output must be valid JSON: ["pkg1", "pkg2", ...] — nothing else.

    Python imports to analyze:
    {cleaned}
    """

    agent = Agent(
        model=Ollama(id="mistral:latest"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,   # 🔑 enforce structured JSON
    )

    try:
        response = agent.run(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        #print("RAW RESPONSE:", response_text)

        packages = json.loads(response_text)
        if not isinstance(packages, list):
            raise ValueError("LLM did not return a JSON list")

        return [pkg.strip() for pkg in packages if isinstance(pkg, str) and pkg.strip()]

    except Exception as e:
        print("Fallback to empty package list due to error:", e)
        return []

def remove_think_tags(text: str) -> str:
    """
    Removes all occurrences of <think>...</think> tags from a string.

    Args:
        text (str): The input string potentially containing <think> tags.

    Returns:
        str: The cleaned string without any <think>...</think> segments.
    """
    cleaned_text = text
    while "<think>" in cleaned_text and "</think>" in cleaned_text:
        start = cleaned_text.find("<think>")
        end = cleaned_text.find("</think>") + len("</think>")
        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
    return cleaned_text

def save_code_to_file(code_str: str, output_dir: str, filename: str, mode="w"):
    """
    Saves code string. If code contains # FILE: markers, splits into multiple files.
    Otherwise saves as single file.
    
    Args:
        code_str (str): The code to save (may contain # FILE: markers)
        output_dir (str): Base directory path
        filename (str): Session filename (used for session folder)
        mode (str): Ignored when splitting files
    """
    import re
    from datetime import datetime
    
    # Extract timestamp from filename or create new one
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"generated_code_{timestamp}")
    
    # Check if code has FILE markers
    file_pattern = re.compile(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', re.MULTILINE)
    file_matches = list(file_pattern.finditer(code_str))
    
    if not file_matches:
        # No markers: save as single file (fallback)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_str)
        return
    
    # Has markers: split into multiple files
    os.makedirs(session_dir, exist_ok=True)
    
    for i, match in enumerate(file_matches):
        file_name = match.group(1)
        start = match.end() + 1  # +1 to skip newline after marker
        
        # Find end: either next FILE marker or end of string
        if i + 1 < len(file_matches):
            end = file_matches[i + 1].start()
        else:
            end = len(code_str)
        
        file_content = code_str[start:end].strip()
        
        # Ensure .py extension
        if not file_name.endswith('.py'):
            file_name += '.py'
        
        file_path = os.path.join(session_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
    
    # Print summary
    print(f"\n✅ Code saved to: {session_dir}")
    print(f"📁 Files created:")
    for file in sorted(os.listdir(session_dir)):
        file_path = os.path.join(session_dir, file)
        size = os.path.getsize(file_path)
        print(f"   • {file} ({size} bytes)")
