#langgraph dev --no-reload --allow-blocking  # to run without reloading and allow blocking operations (to use DeepEval)

# ===========================
# CRITICAL: Path Configuration
# ===========================
import sys
import os
from pathlib import Path

# Get the directory where this script is located
current_script_dir = Path(__file__).parent.absolute()

# Add it to sys.path FIRST, before any other imports
if str(current_script_dir) not in sys.path:
    sys.path.insert(0, str(current_script_dir))

print(f"✅ Added to sys.path: {current_script_dir}")
print(f"✅ Python will search for modules in: {current_script_dir}")



import os
os.environ["DISABLE_NEST_ASYNCIO"] = "1"



# -------------------------
# Core Python & Utilities
# -------------------------
import json                     # For parsing and storing JSON data
import logging                  # Logging and debug output
import re                       # Regular expressions for text processing
import textwrap                 # Formatting text (wrapping)
import asyncio                  # Async operations
import numpy as np              # Numeric arrays and operations
import torch                    # PyTorch (GPU/CPU tensors, deep learning)
torch.cuda.empty_cache()        # Clear CUDA memory if needed
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


# -----------------------------
# Global logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # Show INFO, WARNING, ERROR, CRITICAL messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  
    # Using a safe, standard format that includes timestamp, logger name, level, and message
)

logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)
# Only show WARNING and ERROR from the langgraph_api.server logger to reduce clutter

logger = logging.getLogger(__name__)  
# Use this logger in your module: logger.info(...), logger.warning(...), etc.





# -------------------------
# Type Annotations & Data Models
# -------------------------
from typing import TypedDict
from typing_extensions import Literal
from typing import Annotated
from pydantic import BaseModel, Field  # For structured data validation
from typing import List

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
# CopilotKit / LangGraph Utilities
# -------------------------
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_exit, copilotkit_customize_config



# -------------------------
# Custom Assistant Modules
# -------------------------
from src.assistant.configuration import Configuration
from src.assistant.utils import deduplicate_and_format_sources, format_sources
from src.assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from src.assistant.prompts import (
    query_writer_instructions, summarizer_instructions, reflection_instructions,
    web_search_instructions, web_search_description, web_search_expected_output,
    router_instructions, code_assistant_instructions, academic_summarizer_instructions,
    code_reflection_instructions, code_search_instructions, code_normalization_instructions
)

from helper import (
    load_and_split,
    build_vectorstore,
    extract_packages_from_imports,
    remove_think_tags,
    save_code_to_file
)



# ===== Web Branch Imports =====
from web_branch import (
    generate_query,
    web_research,
    json_parser,
    summarize_sources,
    reflect_on_summary,
    finalize_summary,
    evaluate_summary_with_interrupt,
    route_research
)

# ===== Academic Branch Imports =====
from academic_branch import (
    generate_academic_query,
    academic_research,
    summarize_academic_sources,
    reflect_on_academic_summary,
    finalize_academic_summary,
    evaluate_academic_summary_with_interrupt,
    route_academic_research
)








# Nodes
async def route_question(state: SummaryState, config: RunnableConfig):
    """ Route question to the appropriate node """

    state.research_loop_count = 0

    #print(f"Current state: {state}")

    await copilotkit_emit_message(config, json.dumps({
        "node": "Routing Question",
        "content": "Routing question to the appropriate node..."
    }))
    
    # Customize config to not emit tool calls during routing
    config = copilotkit_customize_config(config, emit_tool_calls=False)

    configurable = Configuration.from_runnable_config(config)

    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")

    # Send system and human messages
    result = await llm_json_mode.ainvoke([ 
        SystemMessage(content=router_instructions),
        HumanMessage(content=f"Input: {state.research_topic}"),
    ])


    print(f"Routing result: {result.content}")



    result_content = json.loads(result.content)

    # Extract the 'response' field from the result content (this is a hashable value)
    response = result_content.get("response")

    # Log the response to ensure it's being extracted correctly
    print(f"Response extracted: {response}")

    # Return state update instead of raw string
    if response == "Academic Source":
        return {"route": "Academic Source"}
    elif response == "Code":
        return {"route": "Code"}
    elif response == "General Web Search":
        return {"route": "General Web Search"}
    else:
        # Fallback to a default route
        return {"route": "General Web Search"}



# ============================================
# UNIVERSAL PARSING UTILITIES
# Use these with ANY agent (nni, snnTorch, code_gen, etc.)
# ============================================

import json
import re
from typing import Dict, Any, Optional


def normalize_code_escapes(text: str) -> str:
    """
    UNIVERSAL: Normalize ALL escape sequences in code string.
    Works with any code string from any source.
    
    Handles both JSON escapes and raw markdown escapes consistently.
    Safe to call multiple times (idempotent).
    
    Args:
        text: Code string potentially with escaped sequences
    
    Returns:
        Cleaned string with actual newlines instead of escaped sequences
    
    Examples:
        normalize_code_escapes("line1\\nline2")  # → "line1\nline2"
        normalize_code_escapes("tab\\there")      # → "tab\there"
    """
    if not isinstance(text, str):
        return text
    
    # Replace common escape sequences
    text = text.replace(r'\\n', '\\n')
    text = text.replace(r'\\t', '\\t')
    text = text.replace(r'\\r', '\\r')
    text = text.replace(r'\\"', '"')
    
    return text


def safe_parse_agent_response(response_text: str, verbose: bool = False) -> Dict[str, Any]:
    """
    UNIVERSAL: Parse ANY agent response with consistent validation and unescaping.
    Works with nni_agent, snnTorch_agent, or any LLM-based agent.
    
    Tries multiple parsing strategies:
    1. JSON parsing (most structured)
    2. Markdown code block extraction
    3. Raw text fallback
    
    Args:
        response_text: Raw response from LLM
        verbose: Print debug information
    
    Returns:
        Guaranteed structure:
        {
            "code": str,           # Complete code with # FILE: markers
            "summary": str,        # Description of generated code
            "files": Dict,         # Extracted files {filename: content}
            "error": str or None,  # Error message if parsing failed
            "parse_method": str    # Which method succeeded (for debugging)
        }
    """
    
    if not response_text or not response_text.strip():
        return {
            "code": "",
            "summary": "",
            "files": {},
            "error": "Empty response",
            "parse_method": "none"
        }
    
    # STEP 1: Strip outer markdown fences universally
    parse_content = response_text.strip()
    
    if '```' in parse_content:
        parts = parse_content.split('```')
        if len(parts) >= 3:
            inner = '```'.join(parts[1:-1])
            lines = inner.split('\\n', 1)
            
            if lines and lines.strip() in ['json', 'python', 'py', 'txt', '']:
                parse_content = lines if len(lines) > 1 else ""
            else:
                parse_content = inner
    
    parse_content = parse_content.strip()
    
    # STEP 2: Try JSON parsing first
    if verbose:
        print("[Parser] ATTEMPT 1: JSON parsing...")
    
    try:
        data = json.loads(parse_content)
        
        if isinstance(data, dict):
            # Map alternative field names to "code"
            if "code" not in data:
                for alt_name in ['script', 'code_content', 'python_code', 'content']:
                    if alt_name in data:
                        data["code"] = data.pop(alt_name)
                        break
            
            # Normalize escapes in code field
            if "code" in data and isinstance(data["code"], str):
                data["code"] = normalize_code_escapes(data["code"])
                if verbose:
                    print(f"[Parser] ✓ JSON parsed, unescaped code ({len(data['code'])} chars)")
            
            # Ensure required fields
            data.setdefault("summary", "Generated code")
            data.setdefault("error", None)
            data.setdefault("files", {})
            data.setdefault("parse_method", "json")
            
            return data
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[Parser] ✗ JSON parse failed: {str(e)[:50]}")
    
    # STEP 3: Extract all code blocks from markdown
    if verbose:
        print("[Parser] ATTEMPT 2: Extracting markdown blocks...")
    
    code_blocks = re.findall(r'```[a-z]*\\n?(.*?)\\n?```', parse_content, re.DOTALL)
    
    if code_blocks:
        unique_blocks = []
        seen = set()
        
        for block in code_blocks:
            normalized = normalize_code_escapes(block.strip())
            
            if normalized and normalized not in seen:
                unique_blocks.append(normalized)
                seen.add(normalized)
        
        if unique_blocks:
            merged_code = '\\n\\n'.join(unique_blocks)
            
            if verbose:
                print(f"[Parser] ✓ Extracted {len(unique_blocks)} blocks ({len(merged_code)} chars)")
            
            return {
                "code": merged_code,
                "summary": f"Extracted {len(unique_blocks)} code blocks from markdown",
                "files": {},
                "error": None,
                "parse_method": "markdown_blocks"
            }
    
    # STEP 4: Try raw text as code
    if verbose:
        print("[Parser] ATTEMPT 3: Using raw text as code...")
    
    code = normalize_code_escapes(parse_content)
    
    if code and len(code.strip()) > 50:
        return {
            "code": code,
            "summary": "Raw response text",
            "files": {},
            "error": None,
            "parse_method": "raw_text"
        }
    
    # STEP 5: Complete failure
    if verbose:
        print("[Parser] ✗ All parsing attempts failed")
    
    return {
        "code": "",
        "summary": "",
        "files": {},
        "error": f"Could not parse response (len={len(response_text)})",
        "parse_method": "failed"
    }


def extract_files_from_code(code: str, verbose: bool = False) -> Dict[str, str]:
    """
    UNIVERSAL: Extract individual files from code with # FILE: markers.
    Works with any code string from any agent.
    
    Handles escaped and unescaped content consistently.
    
    Args:
        code: Code string potentially with # FILE: section markers
        verbose: Print debug information
    
    Returns:
        Dict mapping filename to file content
        If no markers found, returns {"main.py": code}
    
    Example:
        code = '''# FILE: config.py
        import nni
        # FILE: train.py
        def train():
            pass'''
        
        files = extract_files_from_code(code)
        # Returns: {
        #     "config.py": "import nni",
        #     "train.py": "def train():\n    pass"
        # }
    """
    if not code or '# FILE:' not in code:
        return {"main.py": code}
    
    pattern = r'^# FILE:\\s*(\\S+\\.py)\\s*$'
    lines = code.split('\\n')
    
    files = {}
    current_file = None
    current_content = []
    
    for line in lines:
        file_match = re.match(pattern, line)
        
        if file_match:
            # Save previous file
            if current_file and current_content:
                files[current_file] = '\\n'.join(current_content).strip()
            
            # Start new file
            current_file = file_match.group(1)
            current_content = []
        else:
            # Add line to current file
            if current_file is not None:
                current_content.append(line)
    
    # Save last file
    if current_file and current_content:
        files[current_file] = '\\n'.join(current_content).strip()
    
    if verbose:
        print(f"[FileExtractor] Found {len(files)} files: {list(files.keys())}")
    
    return files if files else {"main.py": code}





########################################### CODE GENERATION BRANCH #############################################

# -------------------------
# Custom Embedding Wrapper
# -------------------------

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")  # change to "cpu" if needed

    def embed_documents(self, texts):
        import torch, gc
        batch_size = 8  # You can tune this value depending on your GPU size (try 4 or 2 if you hit OOM)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            gc.collect()
            torch.cuda.empty_cache()
            encoded = self.model.encode(
                batch,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False
            )
            embeddings.extend(encoded)
        gc.collect()
        torch.cuda.empty_cache()
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()




def search_relevant_sources(state: SummaryState, config: RunnableConfig):
    """
    Creation of vectorstore from docs of interest.
    Initializes BOTH SNN and NNI vectorstores with proper caching.
    Uses custom SentenceTransformerEmbeddings class for CUDA memory management.
    """
    
    
    persist_dir_snn = "./chroma_snn_docs"
    persist_dir_nni = "./chroma_nni_docs"
    
    print("\n" + "="*80)
    print("VECTORSTORE INITIALIZATION - SNN + NNI")
    print("="*80 + "\n")
    
    # ✅ Check if both vectorstores already exist and are not empty
    snn_exists = os.path.exists(persist_dir_snn) and os.listdir(persist_dir_snn)
    nni_exists = os.path.exists(persist_dir_nni) and os.listdir(persist_dir_nni)
    
    if snn_exists and nni_exists:
        print(f"✓ SNN vectorstore exists at {persist_dir_snn}")
        print(f"✓ NNI vectorstore exists at {persist_dir_nni}")
        print(f"\n✅ Both vectorstores already initialized (skipping rebuild)")
        print("="*80 + "\n")
        return {"status": "vectorstore_already_exists"}
    
    # ============================================================
    # INITIALIZE EMBEDDING MODEL (CUSTOM CLASS)
    # ============================================================
    
    print("🔧 Loading embedding model (mchochlov/codebert-base-cd-ft)...")
    print("   Using custom SentenceTransformerEmbeddings with CUDA memory management\n")
    
    try:
        embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
        print("   ✓ Embedding model loaded on CUDA with memory optimization\n")
    except Exception as e:
        print(f"   ❌ Error loading embedding model: {e}\n")
        raise
    
    # ============================================================
    # PART 1: SNN VECTORSTORE (Online - from snnTorch docs)
    # ============================================================
    
    if not snn_exists:
        print("-" * 80)
        print("PART 1: CREATING SNN VECTORSTORE (from online documentation)")
        print("-" * 80 + "\n")
        
        try:
            print("📚 Fetching snnTorch documentation from https://snntorch.readthedocs.io/...")
            loader = RecursiveUrlLoader(
                url="https://snntorch.readthedocs.io/en/latest/",
                max_depth=7,
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
            )
            snn_docs = loader.load()
            print(f"   ✓ Loaded {len(snn_docs)} snnTorch documentation pages\n")

                        
            # Split documents
            print("✂️  Splitting documents (chunk_size=512, overlap=50)...")
            doc_splits = load_and_split(snn_docs, chunk_size=512, overlap=50)
            print(f"   ✓ Created {len(doc_splits)} document chunks")
            print(f"   ℹ️  Starting embedding with batching (batch_size=8)...\n")
            
            # Create & persist SNN vectorstore
            print(f"🗄️  Creating Chroma vectorstore (collection: snntorch-docs)...")
            vectorstore_snn = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding_model,
                collection_name="snntorch-docs",
                persist_directory=persist_dir_snn,
                collection_metadata={"hnsw:space": "cosine"}
            )
            vectorstore_snn.persist()
            print(f"   ✓ SNN vectorstore persisted at '{persist_dir_snn}'")
            
        except Exception as e:
            print(f"   ⚠️  Error creating SNN vectorstore: {e}")
            print(f"   ⚠️  Continuing with NNI vectorstore initialization...\n")
    else:
        print("-" * 80)
        print("PART 1: SNN VECTORSTORE - ALREADY EXISTS (skipping)")
        print("-" * 80 + "\n")
    
    # ============================================================
    # PART 2: NNI VECTORSTORE (Local - from esempi_NNI directory)
    # ============================================================
    
    if not nni_exists:
        print("-" * 80)
        print("PART 2: CREATING NNI VECTORSTORE (from esempi_NNI directory)")
        print("-" * 80 + "\n")
        
        # Create NNI API Reference document
        nni_api_ref_doc = Document(
            page_content="""# NNI API REFERENCE - CRITICAL DOCUMENTATION

                ## AlgorithmConfig - Tuner Selection
                AlgorithmConfig(name="Anneal", class_args={"optimize_mode": "maximize"})
                AlgorithmConfig(name="TPE", class_args={"optimize_mode": "maximize"})
                AlgorithmConfig(name="Random", class_args={"optimize_mode": "maximize"})

                ## AlgorithmConfig - Assessor Selection
                AlgorithmConfig(name="Medianstop", class_args={"optimize_mode": "maximize", "start_step": 10})
                AlgorithmConfig(name="NoAssessor")

                ## LocalConfig - GPU Training Service Configuration
                LocalConfig(
                    trial_gpu_number=1,
                    max_trial_number_per_gpu=5,
                    use_active_gpu=True
                )

                ## ExperimentConfig - REQUIRED FIELDS (must all be present)
                - experiment_name: str (name of experiment)
                - experiment_working_directory: str (path to working dir)
                - search_space: dict (JSON format with parameter definitions)
                - tuner: AlgorithmConfig (algorithm configuration)
                - assessor: AlgorithmConfig (early stopping configuration)
                - training_service: LocalConfig (GPU/CPU resource management)
                - max_trial_number: int (maximum number of trials)
                - max_experiment_duration: str (format: "100d", "5h", "30m")
                - trial_concurrency: int (parallel trials)
                - trial_command: str (command to run training script)
                - trial_code_directory: str (directory containing training script)

                ## Key NNI Methods
                experiment = Experiment(config)
                experiment.run(port=8081)
                experiment.stop()

                # Inside training script (main_*.py):
                nni.get_next_parameter()  # Get hyperparameters for this trial
                nni.report_intermediate_result(accuracy)  # Report metrics during training
                nni.report_final_result(best_accuracy)  # Report final result

                ## Search Space Parameter Types
                '_type': 'choice'      → Discrete values: [1, 2, 4, 8, 16]
                '_type': 'quniform'    → Quantized continuous: [min, max, step]
                '_type': 'uniform'     → Continuous: [min, max]
                '_type': 'loguniform'  → Log-scale continuous: [min, max]
                '_type': 'normal'      → Gaussian: [mean, std]
                '_type': 'qnormal'     → Gaussian quantized: [mean, std, step]

                ## Common Hyperparameters for SNN
                neurons_per_pop: choice [2, 4, 8, 16]
                output_pop: choice [1, 2, 4, 8, 16]
                nb_hidden: choice [16, 32, 64]
                alpha_hid: quniform [0, 1, 0.05]
                beta_hid: quniform [0, 1, 0.05]
                alpha_out: quniform [0, 1, 0.05]
                beta_out: quniform [0, 1, 0.05]
                learning_rate: choice [0.0001, 0.0005, 0.001, 0.01]
                reg_l1: quniform [0, 1e-2, 1e-4]
                reg_l2: quniform [0, 1e-4, 1e-6]
                slope: quniform [5, 20, 1]
                batch_size: choice [32, 64, 128]
                """,
            metadata={"source": "NNI_API_REFERENCE", "priority": "HIGHEST"}
        )
        
        print("📂 Loading NNI examples from esempi_NNI directory...")
        
        # Check if directory exists
        if not os.path.exists("esempi_NNI"):
            print("   ❌ esempi_NNI directory NOT found!")
            print("   ℹ️  Expected path: ./esempi_NNI/")
            print("   ℹ️  Creating vectorstore with API reference only...\n")
            nni_docs = [nni_api_ref_doc]
        else:
            print(f"   ✓ esempi_NNI directory found\n")
            
            # Load all files from directory
            try:
                directory_loader = DirectoryLoader(
                    path="esempi_NNI",
                    glob="**/*",
                    show_progress=True
                )
                nni_docs = directory_loader.load()
                print(f"\n   ✓ Loaded {len(nni_docs)} NNI example files from esempi_NNI/")
                
                # List loaded files
                for doc in nni_docs[:4]:
                    source = doc.metadata.get("source", "unknown")
                    print(f"      - {source} ({len(doc.page_content)} chars)")
                if len(nni_docs) > 4:
                    print(f"      - ... and {len(nni_docs) - 4} more files\n")
                else:
                    print()
                
                # Insert API reference at beginning (highest priority)
                nni_docs.insert(0, nni_api_ref_doc)
                print(f"   ✓ Inserted NNI API reference at priority position\n")
                
            except Exception as e:
                print(f"   ⚠️  Error loading directory: {e}")
                print(f"   ℹ️  Using API reference only...\n")
                nni_docs = [nni_api_ref_doc]
        
        # Split NNI documents
        print("✂️  Splitting NNI documents (chunk_size=512, overlap=50)...")
        nni_doc_splits = load_and_split(nni_docs, chunk_size=512, overlap=50)
        print(f"   ✓ Created {len(nni_doc_splits)} document chunks")
        print(f"   ℹ️  Starting embedding with batching (batch_size=8)...\n")
        
        # Create & persist NNI vectorstore
        print(f"🗄️  Creating Chroma vectorstore (collection: nni-docs)...")
        vectorstore_nni = Chroma.from_documents(
            documents=nni_doc_splits,
            embedding=embedding_model,
            collection_name="nni-docs",
            persist_directory=persist_dir_nni,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vectorstore_nni.persist()
        print(f"   ✓ NNI vectorstore persisted at '{persist_dir_nni}'")
        print(f"   ✓ Total documents indexed: {len(nni_doc_splits)}")
        print(f"   ℹ️  GPU memory cleaned up after vectorstore creation\n")
    else:
        print("-" * 80)
        print("PART 2: NNI VECTORSTORE - ALREADY EXISTS (skipping)")
        print("-" * 80 + "\n")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("="*80)
    print("✅ VECTORSTORE INITIALIZATION COMPLETE")
    print("="*80)
    print(f"\n📊 Vectorstore Summary:")
    print(f"   SNN: {persist_dir_snn}/")
    print(f"   NNI: {persist_dir_nni}/")
    print(f"\n🔧 Configuration:")
    print(f"   Embedding model: mchochlov/codebert-base-cd-ft (CUDA optimized)")
    print(f"   Memory management: Batching (8) + Cache clearing enabled")
    print(f"   NNI source: esempi_NNI directory + API reference")
    print(f"\n✨ Status: Both vectorstores ready for agent retrieval!\n")
    
    return {"status": "vectorstore_created"}


def safe_llm_json(content: str):
    """Safely parse JSON with multiple fallback strategies."""
    content = content.strip()
    
    # Remove possible markdown code fences like ```json ... ```
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        content = content.strip()
    
    # Try direct JSON parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON object from within text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        # If everything fails, raise the original error
        raise



def generate_optimized_search_queries(
    user_question: str,
    agent_type: str = "general",  # ← ADD THIS LINE
    num_queries: int = 6,
    model_id: str = "gpt-oss:20b"
) -> list:

    """
    Generate diverse, optimized search queries based on user question.
    
    Args:
        user_question: The original user request
        num_queries: Number of queries to generate (default 10)
        model_id: LLM model to use
        
    Returns:
        list: Optimized search queries ranked by relevance
    """
    
    query_context = {
        "snntorch": "snnTorch library, neural networks, neurons, models, training, loss, backward",
        "nni": "NNI framework, hyperparameter tuning, experiment configuration, dual-file pattern, SearchSpaceUpdater, ExperimentConfig, AlgorithmConfig, search space, _type _value",
        "general": "machine learning, Python, optimization"
    }
    context_hint = query_context.get(agent_type, query_context["general"])


    query_generation_prompt = f"""You are an expert at generating comprehensive search queries for {agent_type.upper()} documentation retrieval.

Generate {num_queries} DIVERSE, SPECIFIC search queries for {agent_type} with focus on: {context_hint}


Your task: Given a user's question, generate {num_queries} DIVERSE, SPECIFIC search queries that will retrieve all relevant documentation needed to answer the question.

Requirements:
1. Each query should target a DIFFERENT aspect of the problem
2. Queries should be specific (include library names, parameter names, concrete concepts)
3. Order queries by importance (most critical first)
4. Avoid duplicate queries or queries that would return the same results
5. Include both broad and narrow/specific queries
6. Cover architecture, implementation, training, and evaluation aspects

User Question: {user_question}

Format: Return ONLY a JSON array of query strings:
["query1", "query2", "query3", ...]

Think about WHAT DOCUMENTATION SECTIONS would be needed to answer this question.
"""

    agent = Agent(
        model=Ollama(id=model_id),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    try:
        response = agent.run(query_generation_prompt)
        print("\n📝 Raw query generation response:\n", response.content[:800], "...\n")
        
        # ✅ Clean and parse JSON safely
        queries = safe_llm_json(response.content)
        
        if not isinstance(queries, list):
            raise ValueError("Expected list of queries")
        
        # Clean and limit to desired number
        queries = [
            q.strip() for q in queries
            if isinstance(q, str) and q.strip()
        ][:num_queries]
        
        print(f"✓ Generated {len(queries)} search queries")
        for i, q in enumerate(queries, 1):
            print(f"  [{i}] {q}")
        
        return queries

    except Exception as e:
        print(f"⚠️ Query generation failed: {e}")
        print("   Falling back to default queries")
        return [
            user_question,
            "snnTorch architecture LIF neurons",
            "PyTorch spiking neural network example",
            "NNI experiment configuration JSON",
            "Neural optimization learning rate tuning",
            "snnTorch backward propagation spike timing",
            "snnTorch dataset preprocessing MNIST",
            "NNI tuner search space Python config",
            "spiking neural networks parameter optimization",
            "PyTorch training loop with NNI integration"
        ]




def rank_queries_by_relevance(
    user_question: str,
    generated_queries: List[str],
    model_id: str = "gpt-oss:20b"
) -> List[str]:
    """
    Rank generated queries by relevance to the user's question.

    Args:
        user_question: Original user question
        generated_queries: List of candidate queries
        model_id: LLM model to use

    Returns:
        list: Ranked queries (highest relevance first)
    """

    ranking_prompt = f"""You are an expert at ranking search query relevance.

User Question: {user_question}

Candidate Queries:
"""
    for i, query in enumerate(generated_queries, 1):
        ranking_prompt += f"{i}. {query}\n"

    ranking_prompt += """
Your task: Rank these queries by RELEVANCE to answering the user's question.

Return ONLY a JSON object:
{
  "ranked_queries": ["most_relevant_query", "second_most_relevant", ...],
  "reasoning": "Brief explanation of ranking"
}

Focus on:
- Core concepts mentioned in question
- Implementation details needed
- Training and evaluation requirements
"""

    # Initialize LLM agent
    agent = Agent(
        model=Ollama(id=model_id),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    try:
        response = agent.run(ranking_prompt)
        print("\n📝 Raw query ranking response:", response)

        content = response.content.strip()

        # Remove markdown code fences if present
        if content.startswith("```"):
            # Remove starting and ending code fences
            content = re.sub(r"^```[\w-]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

        # Parse JSON response
        result = json.loads(content)
        ranked = result.get("ranked_queries", generated_queries)

        print(f"\n📊 Ranked {len(ranked)} queries:")
        for i, q in enumerate(ranked, 1):
            print(f"  [{i}] {q}")

        return ranked

    except Exception as e:
        print(f"⚠️ Ranking failed: {e}")
        return generated_queries





def expand_query_context(
    query: str,
    context_type: str = "general"
) -> str:
    """
    Expand a query with contextual information.
    
    Args:
        query: Base search query
        context_type: "general", "architecture", "training", "evaluation"
        
    Returns:
        str: Expanded query with context
        
    Example:
        >>> expand_query_context("SNN classification", "training")
        "snnTorch SNN classification training loss optimization"
    """
    
    expansions = {
        "architecture": [
            "layer structure",
            "neuron initialization",
            "connection patterns",
            "parameters configuration"
        ],
        "training": [
            "loss function",
            "optimization",
            "backpropagation",
            "gradient flow"
        ],
        "evaluation": [
            "accuracy metrics",
            "performance benchmarks",
            "inference speed"
        ],
        "implementation": [
            "forward pass",
            "backward pass",
            "computational cost",
            "memory usage"
        ]
    }
    
    context_words = expansions.get(context_type, expansions["general"])
    
    # Add random context words to query
    import random
    added_context = random.sample(context_words, min(2, len(context_words)))
    expanded = query + " " + " ".join(added_context)
    
    return expanded


def normalize_code_escapes(text: str) -> str:
    """
    Normalize ALL escape sequences in code string.
    Handles both JSON escapes and raw markdown escapes consistently.

    Args:
        text: Code string potentially with escaped sequences

    Returns:
        Cleaned string with actual newlines instead of escaped sequences
    """
    if not isinstance(text, str):
        return text

    # Pattern 1: Literal backslash sequences in text
    # This catches "\\n" (backslash + n) in raw strings from markdown
    text = text.replace(r'\n', '\n')
    text = text.replace(r'\t', '\t')
    text = text.replace(r'\r', '\r')
    text = text.replace(r'\\"', '\"')

    return text


def safe_parse_nni_response(response_text: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Unified response parser with consistent validation and unescaping.
    Replaces ATTEMPT 1-4 in the original function.

    Tries multiple parsing strategies in order:
    1. JSON parsing (most structured)
    2. Markdown code block extraction
    3. Raw text fallback

    Args:
        response_text: Raw response from LLM
        verbose: Print debug information

    Returns:
        Dict with guaranteed structure:
        {
            "code": str,           # Complete code with # FILE: markers
            "summary": str,        # Description of what was generated
            "files": Dict,         # Extracted files {filename: content}
            "error": str or None,  # Error message if parsing failed
            "parse_method": str    # Which method succeeded (for debugging)
        }
    """

    if not response_text or not response_text.strip():
        return {
            "code": "",
            "summary": "",
            "files": {},
            "error": "Empty response",
            "parse_method": "none"
        }

    # STEP 1: Strip outer markdown fences universally
    parse_content = response_text.strip()

    if '```' in parse_content:
        # Remove code fence markers
        parts = parse_content.split('```')
        if len(parts) >= 3:
            # Assumes: (before)``` (language/content) (content) ```(after)
            inner = '```'.join(parts[1:-1])  # Join all middle parts
            lines = inner.split('\n', 1)

            # Skip language identifier line if present
            if lines and lines[0].strip() in ['json', 'python', 'py', 'txt', '']:
                parse_content = lines[1] if len(lines) > 1 else ""
            else:
                parse_content = inner

    parse_content = parse_content.strip()

    # STEP 2: Try JSON parsing first
    if verbose:
        print("[Parser] ATTEMPT 1: JSON parsing...")

    try:
        data = json.loads(parse_content)

        if isinstance(data, dict):
            # Map alternative field names to "code"
            if "code" not in data:
                for alt_name in ['script', 'code_content', 'python_code', 'content']:
                    if alt_name in data:
                        data["code"] = data.pop(alt_name)
                        break

            # Normalize escapes in code field
            if "code" in data and isinstance(data["code"], str):
                data["code"] = normalize_code_escapes(data["code"])
                if verbose:
                    print(f"[Parser] ✓ JSON parsed, unescaped code ({len(data['code'])} chars)")

            # Ensure required fields
            data.setdefault("summary", "Generated code")
            data.setdefault("error", None)
            data.setdefault("files", {})
            data.setdefault("parse_method", "json")

            return data
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[Parser] ✗ JSON parse failed: {str(e)[:50]}")

    # STEP 3: Extract all code blocks from markdown
    if verbose:
        print("[Parser] ATTEMPT 2: Extracting markdown blocks...")

    code_blocks = re.findall(r'```[a-z]*\n?(.*?)\n?```', parse_content, re.DOTALL)

    if code_blocks:
        # Normalize and deduplicate blocks
        unique_blocks = []
        seen = set()

        for block in code_blocks:
            normalized = normalize_code_escapes(block.strip())

            if normalized and normalized not in seen:
                unique_blocks.append(normalized)
                seen.add(normalized)

        if unique_blocks:
            merged_code = '\n\n'.join(unique_blocks)

            if verbose:
                print(f"[Parser] ✓ Extracted {len(unique_blocks)} blocks ({len(merged_code)} chars)")

            return {
                "code": merged_code,
                "summary": f"Extracted {len(unique_blocks)} code blocks from markdown",
                "files": {},
                "error": None,
                "parse_method": "markdown_blocks"
            }

    # STEP 4: Try raw text as code
    if verbose:
        print("[Parser] ATTEMPT 3: Using raw text as code...")

    code = normalize_code_escapes(parse_content)

    if code and len(code.strip()) > 50:
        return {
            "code": code,
            "summary": "Raw response text",
            "files": {},
            "error": None,
            "parse_method": "raw_text"
        }

    # STEP 5: Complete failure
    if verbose:
        print("[Parser] ✗ All parsing attempts failed")

    return {
        "code": "",
        "summary": "",
        "files": {},
        "error": f"Could not parse response (len={len(response_text)})",
        "parse_method": "failed"
    }


def extract_files_from_code(code: str, verbose: bool = False) -> Dict[str, str]:
    """
    Extract individual files from code with # FILE: markers.
    Handles escaped and unescaped content consistently.

    Args:
        code: Code string potentially with # FILE: section markers
        verbose: Print debug information

    Returns:
        Dict mapping filename to file content
    """
    if not code or '# FILE:' not in code:
        return {"main.py": code}

    # Pattern to extract FILE sections
    pattern = r'^# FILE:\s*(\S+\.py)\s*$'
    lines = code.split('\n')

    files = {}
    current_file = None
    current_content = []

    for line in lines:
        file_match = re.match(pattern, line)

        if file_match:
            # Save previous file
            if current_file and current_content:
                files[current_file] = '\n'.join(current_content).strip()

            # Start new file
            current_file = file_match.group(1)
            current_content = []
        else:
            # Add line to current file
            if current_file is not None:
                current_content.append(line)

    # Save last file
    if current_file and current_content:
        files[current_file] = '\n'.join(current_content).strip()

    if verbose:
        print(f"[FileExtractor] Found {len(files)} files: {list(files.keys())}")

    return files if files else {"main.py": code}



# ------------------ Specialized Agent: snnTorch_agent ------------------
def snnTorch_agent(question: str, config: RunnableConfig, state: SummaryState):
    """
    Enhanced snnTorch agent with LLM-guided search query generation,
    multi-pass retrieval with ranked queries, and robust JSON parsing.
    """


    print("\n" + "="*80)
    print("SNNTORCH AGENT - LLM-OPTIMIZED MULTI-PASS RETRIEVAL")
    print("="*80 + "\n")

    # STEP 1: Initialize embedding model & vectorstore
    print("🔧 Loading embedding model and vectorstore...")
    try:
        embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
    except Exception:
        embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="snntorch-docs",
        persist_directory="./chroma_snn_docs"
    )
    print("   ✓ Vectorstore loaded\n")

    # --- Helper: Safe JSON extraction ---
    def safe_llm_json(content: str):
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
        return json.loads(content)

    # STEP 2: Generate optimized multi-strategy search queries
    print("🧠 Generating optimized search queries from LLM...")

    generate_prompt = f"""You are an expert at generating comprehensive search queries for the snnTorch library documentation.

User Question:
{question}

Generate 6 diverse, focused search queries covering architecture, parameters, training, modules, and usage patterns.

Return ONLY a JSON array of query strings:
["query1", "query2", ..., "query10"]
"""

    generate_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    try:
        gen_response = generate_agent.run(generate_prompt)
        queries = safe_llm_json(gen_response.content)
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("Empty or invalid query list generated")
    except Exception as e:
        print(f"⚠️ Query generation failed: {e}")
        queries = [
            question,
            "snnTorch network architecture LIF neurons",
            "snnTorch encoder decoder spike",
            "snnTorch training loss backward",
            "snnTorch recurrent network state",
            "snnTorch forward pass parameters",
            "snnTorch input data preprocessing",
            "snnTorch model evaluation accuracy",
            "snnTorch GPU CUDA performance",
            "snnTorch examples tutorials"
        ]

    # STEP 3: Rank queries by relevance
    print("🧠 Ranking generated queries by relevance...")

    rank_prompt = f"""You are an expert at ranking search query relevance.

User Question:
{question}

Candidate Queries:
"""
    for idx, q in enumerate(queries, 1):
        rank_prompt += f"{idx}. {q}\n"

    rank_prompt += """
Your task: Rank these queries by RELEVANCE to answering the user's question.

Return ONLY a JSON object:
{
  "ranked_queries": ["most_relevant_query", "second_most_relevant", ...],
  "reasoning": "Brief explanation of ranking"
}

Focus on:
- Core concepts mentioned in question
- Implementation details needed
- Training and evaluation requirements
"""

    rank_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    try:
        rank_response = rank_agent.run(rank_prompt)
        rank_result = safe_llm_json(rank_response.content)
        ranked_queries = rank_result.get("ranked_queries", queries)
        reasoning = rank_result.get("reasoning", "")
        print(f"\n📊 Ranked Queries (reasoning: {reasoning}):")
        for i, rq in enumerate(ranked_queries, 1):
            print(f"  [{i}] {rq}")
    except Exception as e:
        print(f"⚠️ Ranking failed: {e}")
        ranked_queries = queries

    # STEP 4: Multi-pass retrieval with ranked queries
    print("\n🔍 Executing multi-pass retrieval with ranked queries:")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    snn_context = ""
    retrieved_docs = {}
    total_docs = 0

    for i, query in enumerate(ranked_queries, 1):
        rank_score = 1.0 - (i - 1) * 0.05  # simple decay weighting
        docs = retriever.get_relevant_documents(query)
        retrieved_docs[query] = docs
        total_docs += len(docs)
        print(f"   [{i}] Query: '{query[:50]}...' → {len(docs)} docs (weight {rank_score:.2f})")
        for j, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get("source", "unknown")
            snn_context += f"[Query {i}.{j} - {source} - weight {rank_score:.2f}]\n{doc.page_content[:600]}\n---\n"

    print(f"\n   Total retrieved: {total_docs} documents")
    print(f"   Context size: {len(snn_context):,} characters\n")

    # # STEP 5: Extract patterns for prompt conditioning
    # patterns = {
    #     "classes": list(set(re.findall(r"class\s+(\w+)\s*[:\(]", snn_context))),
    #     "functions": list(set(re.findall(r"def\s+(\w+)\s*\(", snn_context)))[:10],
    #     "modules": list(set(re.findall(r"(?:import|from)\s+([\w\.]+)", snn_context))),
    # }

    # print(f"   ✓ Classes: {len(patterns['classes'])} - {patterns['classes'][:5]}")
    # print(f"   ✓ Functions: {len(patterns['functions'])} - {patterns['functions'][:5]}")
    # print(f"   ✓ Modules: {len(patterns['modules'])} - {patterns['modules'][:5]}\n")

    # STEP 6: Build prompt with ranked context
    prompt = f"""You are a snnTorch code generation assistant.

USER REQUEST: {question}

RANKED DOCUMENTATION (most relevant first):

{snn_context[:8000]}


OUTPUT FORMAT (CRITICAL):
Structure your code with # FILE: markers:

# FILE: filename.py
[code]

# FILE: another.py
[code]

Rules:
- Each file starts: # FILE: filename.py
- Put code after marker
- NO markdown fences (no ```)
- NO visual separators

Return JSON: {{"code": "...", "confidence": 0.95, "queries_used": 6}}
"""

    # STEP 7: Generate code with LLM
    print("🤖 Calling LLM for code generation...")
    gen_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    response = gen_agent.run(prompt)
    print("   ✓ LLM generation complete\n")

    # STEP 8: Parse and return
    print("Raw LLM response:\n", response.content, "...\n")
    print("✓ Parsing response...")
    
    parse_result = safe_parse_agent_response(response.content, verbose=True)
    
    if parse_result["error"] is not None:
        print(f"❌ Parsing failed: {parse_result['error']}")
        return {
            "code": "",
            "confidence": 0.0,
            "queries_used": len(ranked_queries) if 'ranked_queries' in locals() else 0,
            "error": parse_result["error"]
        }
    
    code = parse_result["code"]
    files = extract_files_from_code(code, verbose=True)
    
    print(f"✓ Parsed via: {parse_result['parse_method']}")
    print(f"✓ Code length: {len(code):,} chars")
    
    return {
        "code": code,
        "files": files,
        "summary": parse_result.get("summary", "Generated code"),
        "confidence": 0.95,
        "queries_used": len(ranked_queries) if 'ranked_queries' in locals() else 0,
        "error": None
    }










# ------------------ Specialized Agent: nni_agent ------------------

def nni_agent(question: str, config: RunnableConfig, state: SummaryState) -> Dict[str, Any]:
    """
    Enhanced NNI Agent with unified parsing for consistent response handling.

    Replaces the multi-attempt parsing logic with a single robust parser
    that handles all escape sequence combinations.

    Args:
        question: User question
        config: Runnable configuration
        state: Current state with code, files, summary

    Returns:
        Dict with:
        - code: String with all Python code
        - files: Dict mapping filenames to content
        - summary: Description of what was generated
        - error: Error message if failed, None if successful
    """

    print("=" * 80)
    print("NNI AGENT - UNIFIED PARSING (FIXED)")
    print("=" * 80)

    # STEP 1: Initialize vectorstore
    print("[NNI] Initializing vectorstore...")
    try:
        embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
    except Exception:
        embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name="nni-docs",
            persist_directory=".chroma/nni_docs"
        )
        print("[NNI] ✓ Vectorstore initialized")
    except Exception as e:
        print(f"[NNI] ⚠️ Warning: {e}")
        vectorstore = None

    # STEP 2: Generate search queries
    print("[NNI] Generating optimized search queries...")
    search_queries = generate_optimized_search_queries(
        user_question=question,
        agent_type="nni",
        num_queries=6,
        model_id="gpt-oss:20b"
    )

    # STEP 3: Rank queries
    print("[NNI] Ranking queries by relevance...")
    ranking_prompt = f"""You are an expert at ranking search query relevance.

Question: {question}

Candidate Queries:
{chr(10).join(f"{i}. {q}" for i, q in enumerate(search_queries, 1))}

Your task: Rank these queries by RELEVANCE to answering the user's question.
Return ONLY a JSON object:
{{
    "ranked_queries": ["most_relevant_query", "second_most_relevant", ...],
    "reasoning": "Brief explanation of ranking"
}}

Focus on:
- Core concepts mentioned in question
- Implementation details needed
- Training and evaluation requirements
"""

    rank_agent = Agent(model=Ollama(id="gpt-oss:20b"), tools=[], show_tool_calls=False)

    ranked_queries = search_queries
    try:
        rank_response = rank_agent.run(ranking_prompt)
        rank_content = rank_response.content.strip()

        # Clean markdown if present
        if rank_content.startswith("```"):
            rank_content = re.sub(r"```[a-z]*", "", rank_content)
            rank_content = re.sub(r"```", "", rank_content)
            rank_content = rank_content.strip()

        rank_result = json.loads(rank_content)
        ranked_queries = rank_result.get("ranked_queries", search_queries)
        print(f"[NNI] ✓ Ranked {len(ranked_queries)} queries")
    except Exception as e:
        print(f"[NNI] ⚠️ Ranking failed: {e}")

    # STEP 4: Multi-pass retrieval
    print("[NNI] Performing multi-pass retrieval...")
    nni_context = ""

    if vectorstore:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            retrieved_docs = {}
            total_docs = 0

            for i, query in enumerate(ranked_queries, 1):
                rank_score = 1.0 - (i - 1) * 0.05

                try:
                    docs = retriever.get_relevant_documents(query)
                except Exception:
                    docs = []

                retrieved_docs[query] = docs
                total_docs += len(docs)
                print(f"[NNI] Query {i}: {query[:50]}... → {len(docs)} docs (weight: {rank_score:.2f})")

                for j, doc in enumerate(docs[:5], 1):
                    source = doc.metadata.get("source", "unknown")
                    nni_context += f"Query {i}.{j} - {source} - weight {rank_score:.2f}\n{doc.page_content[:600]}\n---\n"

            print(f"[NNI] ✓ Retrieved {total_docs} documents")
            print(f"[NNI] ✓ Context size: {len(nni_context):,} characters")
        except Exception as e:
            print(f"[NNI] ⚠️ Retrieval failed: {e}")

    # STEP 5: Build LLM prompt
    print("[NNI] Building generation prompt...")

    prompt = f"""
CRITICAL FORMAT REQUIREMENT:
Your response MUST be EXACTLY ONE valid JSON object with this structure:
{{
    "code": "# FILE: config.py\n...\n# FILE: train.py\n...",
    "summary": "Brief description of generated code"
}}

RULES:
1. EXACTLY 2 fields: code and summary
2. NO markdown code blocks, NO explanations outside JSON
3. ALL code in code field with # FILE: markers
4. Must be parseable with json.loads()
5. Each file section starts with # FILE: filename.py

---

You are an NNI configuration expert. Your task:

{question}

CONTEXT (ranked by relevance):
{nni_context[:5000]}

Generate complete NNI configuration with:
- config.py: Search space definition, experiment config, tuner setup
- train.py: Training loop with NNI integration, metrics reporting
- utils.py (if needed): Helper functions

Format as pure JSON ONLY. No explanations or markdown.
"""

    # STEP 6: Call LLM
    print("[NNI] Calling LLM for code generation...")

    code_agent = Agent(model=Ollama(id="gpt-oss:20b"), tools=[], show_tool_calls=False, use_json_mode=True)

    try:
        response = code_agent.run(prompt)
        response_text = response.content
        print(response_text)
        print(f"[NNI] ✓ LLM response received ({len(response_text)} chars)")
    except Exception as e:
        print(f"[NNI] ❌ LLM call failed: {e}")
        return {
            "code": "",
            "files": {},
            "summary": "",
            "error": str(e)
        }

    # STEP 7-8: UNIFIED PARSING WITH FIXED LOGIC
    print("[NNI] Parsing response with unified parser...")

    parse_result = safe_parse_nni_response(response_text, verbose=True)

    # Check for errors
    if parse_result["error"] is not None:
        print(f"[NNI] ❌ Parsing failed: {parse_result['error']}")
        return {
            "code": "",
            "files": {},
            "summary": "",
            "error": parse_result["error"]
        }

    code = parse_result["code"]
    summary = parse_result.get("summary", "Generated NNI configuration")
    parse_method = parse_result.get("parse_method", "unknown")

    print(f"[NNI] ✓ Parsed via: {parse_method}")
    print(f"[NNI] ✓ Code length: {len(code):,} chars")
    print(f"[NNI] ✓ Summary: {summary}")

    # STEP 9: Validate schema
    print("[NNI] Validating response schema...")

    if not isinstance(code, str) or len(code.strip()) == 0:
        return {
            "code": "",
            "files": {},
            "summary": summary,
            "error": "Code field empty or invalid"
        }

    # STEP 10: Extract files consistently
    print("[NNI] Extracting files from code...")

    files = extract_files_from_code(code, verbose=True)

    print(f"[NNI] ✓ Extracted {len(files)} files: {list(files.keys())}")

    # STEP 11: Final validation
    print("[NNI] Performing final validation...")

    if "# FILE:" not in code:
        code = f"# FILE: config.py\n{code}"
        print("[NNI] ⚠️ Added FILE marker to code")

    # Ensure files are properly formed
    file_markers = re.findall(r'^# FILE:\s*(\S+\.py)\s*$', code, re.MULTILINE)

    if not file_markers:
        print("[NNI] ⚠️ Warning: Could not find FILE markers")
    else:
        print(f"[NNI] ✓ Found {len(set(file_markers))} unique files")

    # STEP 12: Return result
    print("=" * 80)
    print("NNI AGENT COMPLETE - SUCCESS")
    print("=" * 80)
    print(f"[NNI] Files: {len(files)} ({', '.join(files.keys())})")
    print(f"[NNI] Code: {len(code):,} characters")
    print(f"[NNI] Parse method: {parse_method}")

    return {
        "code": code,
        "files": files,
        "summary": summary,
        "error": None
    }







# ------------------ General Code Generation Function ------------------
def generate_code(config: RunnableConfig, state: SummaryState):
    """
    Orchestrator with proper agent communication and validation.
    
    Fixed issues:
    - ✅ Correct agent output schema documentation
    - ✅ Proper error field validation
    - ✅ Intelligent integration (no redundant LLM calls)
    - ✅ FILE marker validation
    """
    
    print("---GENERATING CODE SOLUTION---")
    print("--- PHASE 1: TOOL SELECTION AND EXECUTION ---")
    
    question = state.research_topic
    if isinstance(question, list):
        question_str = " ".join(question)
    else:
        question_str = str(question)
    
    # Define available tools with CORRECT output format documentation
    available_tools = [
        {
            "name": "snnTorch_agent",
            "function": lambda q: snnTorch_agent(q, config, state),
            "description": "Generates SNN model code using snnTorch library"
        },
        {
            "name": "nni_agent",
            "function": lambda q: nni_agent(q, config, state),
            "description": "Generates NNI experiment configuration and setup code"
        }
    ]
    
    tool_selection_prompt = """You are a code generation orchestrator that decides which specialized tools to invoke.

Your task:
- Analyze the user request to determine which tools are needed
- For each tool, provide a specific, focused query
- Each tool will return code formatted with # FILE: markers
- Respond with ONLY a JSON object containing tool calls

Available tools:
1. snnTorch_agent: Generates SNN model code
- Returns dict with:
  - "code": String with all Python code, includes # FILE: markers for each module
  - "files": Dict mapping filenames to content
  - "summary": Description of what was generated
  - "error": null if successful, error message if failed
- Each file section starts with: # FILE: filename.py
- Output example:
    # FILE: model.py
    [SNN model code]
    
    # FILE: utils.py
    [helper functions]

2. nni_agent: Generates NNI experiment configuration
- Returns dict with:
  - "code": String with all Python code, includes # FILE: markers for each component
  - "files": Dict mapping filenames to content
  - "summary": Description of what was generated
  - "error": null if successful, error message if failed
- Each file section starts with: # FILE: filename.py
- Output example:
    # FILE: config.py
    [NNI configuration]
    
    # FILE: train.py
    [training loop]

Response format (ONLY valid JSON, no markdown):
{
  "tool_calls": [
    {
      "name": "snnTorch_agent",
      "query": "specific question for SNN model code",
      "expected_files": ["model.py", "utils.py"]
    },
    {
      "name": "nni_agent",
      "query": "specific question for NNI setup",
      "expected_files": ["config.py", "train.py"]
    }
  ]
}

Critical Rules:
- Only include tools that are needed
- Each query should be focused and specific
- Do NOT generate code yourself - only decide which tools to call
- Output ONLY valid JSON (no markdown code blocks, no explanations)

IMPORTANT: Each tool will return a dict. You don't need to specify format_requirement - just identify which tools to call!
"""
    
    prompt = (
        f"Context (previous code):\n{state.code}\n\n"
        f"User request:\n{question_str}\n\n"
        f"Instructions: {tool_selection_prompt}"
    )
    
    # Get tool selection from orchestrator
    response = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=True,
        use_json_mode=True,
    ).run(prompt)
    
    print("ORCHESTRATOR SPECIALIZED AGENTS SELECTION:\n", response.content)
    
    # ============================================================
    # PARSE AND EXECUTE TOOL CALLS (FIXED)
    # ============================================================
    
    try:
        response_data = json.loads(response.content)
        tool_calls = response_data.get("tool_calls", [])
        
        if not tool_calls:
            print("WARNING: No tool calls found")
            return "Error: No tools were selected"
            
        print(f"Tools selected: {[call['name'] for call in tool_calls]}")
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse orchestrator response: {e}")
        return "Error in response parsing"
    
    # Execute tools and collect outputs
    tool_outputs = {}
    failed_tools = []

    for call in tool_calls:
        tool_name = call.get("name")
        query = call.get("query")
        expected_files = call.get("expected_files", [])
        
        if not tool_name or not query:
            print(f"WARNING: Skipping invalid tool call: {call}")
            continue
        
        print(f"\n📋 Executing tool: {tool_name}")
        print(f"   Query: {query[:80]}...")
        if expected_files:
            print(f"   Expected output files: {expected_files}")
        
        # ============================================================
        # EXECUTE THE TOOL
        # ============================================================
        
        result = None
        if tool_name == "snnTorch_agent":
            result = snnTorch_agent(query, config, state)
        elif tool_name == "nni_agent":
            result = nni_agent(query, config, state)
        else:
            print(f"   ❌ Unknown tool: {tool_name}")
            failed_tools.append((tool_name, "Unknown tool"))
            continue
        
        # ============================================================
        # VALIDATE RESULT FORMAT (FIXED)
        # ============================================================
        
        # Check if result is dict
        if not isinstance(result, dict):
            print(f"   ❌ Tool returned non-dict: {type(result)}")
            failed_tools.append((tool_name, f"Non-dict result: {type(result)}"))
            continue
        
        # CRITICAL FIX: Check error field FIRST
        error = result.get("error")
        if error is not None:
            error_msg = str(error)
            print(f"   ❌ Tool failed with error: {error_msg}")
            print(f"   ⚠️  Skipping this tool output")
            failed_tools.append((tool_name, error_msg))
            continue  # Skip this tool - it failed
        
        # ============================================================
        # EXTRACT AND VALIDATE CODE
        # ============================================================
        
        code = result.get("code", "")
        
        if not code or len(code.strip()) < 50:
            print(f"   ❌ Tool returned insufficient code ({len(code)} chars)")
            failed_tools.append((tool_name, "Insufficient code"))
            continue
        
        # CRITICAL FIX: Validate FILE markers BEFORE accepting code
        import re
        file_markers = re.findall(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', code, re.MULTILINE)
        
        if len(file_markers) == 0:
            print(f"   ❌ CRITICAL: Code has no # FILE: markers")
            print(f"   Code preview: {code[:200]}...")
            failed_tools.append((tool_name, "No FILE markers"))
            continue
        
        print(f"   ✓ Code length: {len(code):,} characters")
        print(f"   ✓ Files detected: {file_markers}")
        print(f"   ✓ Files found: {len(file_markers)} ({', '.join(file_markers)})")
        
        # Get summary
        summary = result.get("summary", "Generated code")
        print(f"   ✓ Summary: {summary[:100]}...")
        
        # Get files dict for logging
        files = result.get("files", {})
        print(f"   ✓ Output dict has {len(files)} files")
        
        # SUCCESS: Store this tool's output
        tool_outputs[tool_name] = {
            "code": code,
            "files": files,
            "summary": summary
        }
        print(f"   ✅ ACCEPTED: {tool_name} output ready for integration")

    # ============================================================
    # CHECK IF WE GOT ANY VALID OUTPUTS
    # ============================================================
    
    if not tool_outputs:
        print(f"\n❌ ERROR: No code was generated from any tool")
        if failed_tools:
            print(f"Failed tools:")
            for tool_name, reason in failed_tools:
                print(f"  - {tool_name}: {reason}")
        return "Error: No code generated from tools"

    print(f"\n✅ Collected {len(tool_outputs)} valid tool outputs")
    if failed_tools:
        print(f"⚠️  {len(failed_tools)} tools failed and were skipped")

    # ============================================================
    # PHASE 2: CODE INTEGRATION (SIMPLIFIED & FIXED)
    # ============================================================
    
    print(f"\n--- PHASE 2: CODE INTEGRATION ---")
    print(f"Integrating {len(tool_outputs)} code components...")
    
    # FIXED: If only one tool, use its output directly (no re-running LLM)
    if len(tool_outputs) == 1:
        tool_name = list(tool_outputs.keys())[0]
        tool_output = tool_outputs[tool_name]
        integrated_code = tool_output["code"]
        
        print(f"\n📦 Single tool output detected")
        print(f"   Tool: {tool_name}")
        print(f"   Summary: {tool_output['summary'][:100]}...")
        print(f"   Using output directly (no merging needed)")
        
    else:
        # FIXED: For multiple tools, intelligently merge WITHOUT re-running LLM
        print(f"\n📦 Multiple tool outputs - merging intelligently...")
        
        # Extract all files from all tools
        all_files_dict = {}  # filename -> {tool_name: ..., content: ...}
        
        for tool_name, tool_output in tool_outputs.items():
            code = tool_output["code"]
            files = tool_output.get("files", {})
            
            print(f"\n   From {tool_name}:")
            
            # Extract files using regex
            file_sections = re.findall(
                r'^# FILE:\s*(\S+\.py)\s*\n((?:(?!^# FILE:).)*)',
                code,
                re.MULTILINE | re.DOTALL
            )
            
            for filename, file_content in file_sections:
                file_content = file_content.strip()
                
                if filename not in all_files_dict:
                    all_files_dict[filename] = {
                        "tool_name": tool_name,
                        "content": file_content
                    }
                    print(f"      Added: {filename} ({len(file_content):,} chars)")
                else:
                    # File already exists - keep the first one (from higher priority tool)
                    existing_tool = all_files_dict[filename]["tool_name"]
                    print(f"      Skipped duplicate: {filename} (already from {existing_tool})")
        
        # Define file priority order for proper sequencing
        file_priority = {
            'config.py': 1,
            'search_space.json': 2,
            'model.py': 3,
            'utils.py': 4,
            'train.py': 5,
            'main.py': 6,
            'experiment.py': 7,
        }
        
        # Sort files by priority and concatenate
        sorted_files = sorted(
            all_files_dict.items(),
            key=lambda x: file_priority.get(x[0], 999)
        )
        
        integrated_code = ""
        for filename, file_info in sorted_files:
            integrated_code += f"# FILE: {filename}\n{file_info['content']}\n\n"
        
        integrated_code = integrated_code.strip()
        
        print(f"\n   ✓ Merged {len(all_files_dict)} files in priority order")

    # ============================================================
    # VALIDATION: Check integration output format
    # ============================================================

    print("\n" + "="*80)
    print("INTEGRATION OUTPUT VALIDATION")
    print("="*80)

    # Extract all file markers
    file_markers = re.findall(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', integrated_code, re.MULTILINE)

    print(f"\n📊 Files in output: {len(file_markers)}")
    for i, fname in enumerate(file_markers, 1):
        print(f"   {i}. {fname}")

    # Validation checks
    checks = {
        "has_file_markers": len(file_markers) > 0,
        "files_have_py_extension": all(f.endswith('.py') for f in file_markers),
        "unique_filenames": len(file_markers) == len(set(file_markers)),
        "has_code_file": any(f.lower() in ['model.py', 'config.py', 'main.py'] for f in file_markers),
        "starts_with_file_marker": integrated_code.lstrip().startswith("# FILE:"),
    }

    passed = sum(1 for v in checks.values() if v)
    score = int((passed / len(checks)) * 100)

    print(f"\n✓ Validation score: {score}% ({passed}/{len(checks)} checks)")

    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")

    # Fix common issues
    if len(file_markers) == 0:
        print("\n⚠️  CRITICAL: No # FILE: markers found!")
        print("   This should not happen - tools should provide them!")
        integrated_code = f"# FILE: main.py\n{integrated_code}"

    if "```" in integrated_code:
        print("\n⚠️  WARNING: Found markdown code fences")
        integrated_code = re.sub(r"```[^`]*\n?", "", integrated_code)

    print(f"\n✅ Final output: {len(integrated_code):,} characters")
    print(f"✅ Ready for execution: {len(file_markers)} files")
    print("="*80 + "\n")

    # Save to state
    state.code = integrated_code
    return integrated_code






## Function to generate code
def generate(state: SummaryState, config: RunnableConfig):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")
    #print(f"Current state: {state}")

    # State
    messages = state.messages
    iterations = state.code_iterations

    # Solution
    code = generate_code(config, state)

    
    # ===== COMPOSER AGENT (MINIMAL) =====
    print("--- PHASE 3: CODE COMPOSITION AND FIXING ---")
    
    # STEP 1: Build prompt
    composer_prompt = f"""You are a code composer. Fix this Python code to make it executable.
    
CODE:
{code}

Tasks:
1. Fix syntax errors
2. Ensure imports work
3. Wire connections between files
4. Return ONLY valid JSON:

{{"code": "[COMPLETE fixed code with # FILE: markers - use REGULAR quotes, NO backslashes]"}}

CRITICAL: Code inside "code" field should be raw Python, not escaped!


"""
    
    # STEP 2: Call agent
    composer_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )
    response = composer_agent.run(composer_prompt)
    
    # STEP 3: Extract code
    import json
    try:
        result = json.loads(response.content)
        composed_code = result.get("code", code)
    except:
        composed_code = code

    output_directory = "./generated_code"
    #output_filename = "latest_generated_code.py"  

    # Construct unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_code_{timestamp}.py"

    # To overwrite the file each time:
    save_code_to_file(composed_code, output_directory, output_filename, mode="w")

    state.code = composed_code
    #state.fixed_code = code # save for later sandboxing

    print("CODE SOLUTION:\n", composed_code, "...\n")
    

    

# def generate_code(question: str, config: RunnableConfig, state: SummaryState):


#     print("--- GENERATING CODE ---")

#     # -------------------------
#     # 1️⃣ Web Search for User URLs
#     # -------------------------
#     # Use search tools
#     duck_tool = DuckDuckGoTools(fixed_max_results=3)
#     google_tool = GoogleSearchTools(proxy=None)  # or proxy="http://your_proxy:port"
#     baidusearch_tool = BaiduSearchTools(fixed_max_results=3)

#     search_agent = Agent(
#         model=Ollama(id="qwen3:latest"),
#         tools=[google_tool],
#         show_tool_calls=True,
#         markdown=False
#     )

#     # Format search query
#     search_query = code_search_instructions.format(research_topic=state.research_topic)
#     search_response = search_agent.run(search_query)
#     #print("RAW SEARCH RESPONSE:\n", search_response)

#     # Clean up <think> tags from Deepseek/LLM responses
#     content = search_response.content
#     content = remove_think_tags(content)
    

#     # Parse URLs returned by the search
#     try:
#         data = json.loads(content or "{}")
#         urls = data.get("urls", [])
#     except json.JSONDecodeError:
#         urls = []
#     print(f"User URLs: {urls}")

#     # Load content from user-specified URLs
#     url_docs = []
#     for url in urls:
#         url_docs.extend(WebBaseLoader(url).load())
#     url_context = "\n\n".join([doc.page_content for doc in url_docs])

#     # -------------------------
#     # 2️⃣ Load SNN Documentation Vectorstore
#     # -------------------------
#     embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
#     #embedding_model = SentenceTransformerEmbeddings("nomic-ai/CodeRankEmbed")

#     # Load persisted vectorstore
#     vectorstore = Chroma(
#         embedding_function=embedding_model,
#         collection_name="snntorch-docs",
#         persist_directory="./chroma_snn_docs"
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     # Retrieve relevant SNN docs
#     snn_relevant = retriever.get_relevant_documents(state.research_topic)
#     snn_context = "\n\n".join([doc.page_content for doc in snn_relevant])

#     # -------------------------
#     # 3️⃣ Combine contexts
#     # -------------------------
#     combined_context = snn_context + "\n\n" + url_context

#     # -------------------------
#     # 4️⃣ Construct prompt for code generation
#     # -------------------------
#     code_agent = Agent(
#         model=Ollama(id="gpt-oss:20b"),
#         tools=[],
#         show_tool_calls=False,
#         use_json_mode=True,
#     )

#     prompt = (
#     code_assistant_instructions
#     + "\nBased on the following context, generate the code that satisfies the question:\n"
#     + "Context:\n" + combined_context
#     + "\nQuestion:\n" + state.research_topic
#     + "\nIf the question asks for a modification to previously generated code, "
#       "return the ENTIRE codebase again, with the requested modification fully integrated. "
#       "Do NOT output only the new fragment—always output the complete updated code.\n"
#     + "Here is the previously generated code:\n"
#     + state.code
#     )


#     # Run the code generation agent
#     code_response = code_agent.run(prompt)
#     content = code_response.content
#     code = ""

#     try:
#         content = json.loads(content or "{}")
#         code = content.get("code", "")
#     except json.JSONDecodeError:
#         code = "Error: Unable to parse code response."
        
    

#     return code
    






    # messages += [
    #     (
    #         "assistant",
    #         f"Here is my attempt to solve the problem:\n\n"
    #         f"Code:\n{code}\n```"
           
    #     )
    # ]


    # Increment
    #state.code_iterations = state.code_iterations + 1
    return {"code": code} #, "user_feedback": state.user_feedback






# ============================================================================
# E2B INTEGRATION: Multi-File Code Execution in Sandbox
# ============================================================================

from typing import Dict, List, Optional


def parse_multifile_code(code_str: str) -> Dict[str, str]:
    """
    Parse code string with # FILE: markers into separate files.
    
    Args:
        code_str: Generated code with # FILE: markers
        
    Returns:
        dict: {filename: content} mapping
    """
    file_pattern = re.compile(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', re.MULTILINE)
    file_matches = list(file_pattern.finditer(code_str))
    
    if not file_matches:
        return {"main.py": code_str}
    
    files = {}
    
    for i, match in enumerate(file_matches):
        filename = match.group(1)
        
        if not filename.endswith('.py'):
            filename = filename + '.py'
        
        start = match.end() + 1
        end = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(code_str)
        
        file_content = code_str[start:end].rstrip()
        
        if file_content.strip():
            files[filename] = file_content
    
    return files


def detect_main_file(files: Dict[str, str], hint: Optional[str] = None) -> str:
    """
    Auto-detect the main/entry-point file.
    
    Args:
        files: dict of {filename: content}
        hint: optional hint for main file name
        
    Returns:
        str: filename of main entry point
    """
    if hint and hint in files:
        return hint
    
    priority_names = ["main.py", "train.py", "experiment.py"]
    for name in priority_names:
        if name in files:
            return name
    
    keywords = ['main', 'train', 'run', 'execute']
    for filename in files.keys():
        for keyword in keywords:
            if keyword in filename.lower():
                return filename
    
    return list(files.keys())


def extract_packages_from_imports(imports_str: str) -> List[str]:
    """
    Extract package names from import statements.
    
    Args:
        imports_str: String containing import statements
        
    Returns:
        list: Package names to install
    """
    pattern = r"(?:^import|^from)\s+(\w+)"
    matches = re.findall(pattern, imports_str, re.MULTILINE)
    
    pip_mapping = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'snntorch': 'snntorch',
    }
    
    builtin = {
        'os', 'sys', 'json', 're', 'datetime', 'math', 'random',
        'collections', 'itertools', 'functools', 'pathlib', 'typing',
        'subprocess', 'shutil', 'tempfile', 'hashlib', 'uuid'
    }
    
    packages = []
    for module in matches:
        if module not in builtin:
            package = pip_mapping.get(module, module)
            if package not in packages:
                packages.append(package)
    
    return packages


def upload_and_execute_in_e2b(
    code_str: str,
    main_file: Optional[str] = None,
    install_packages: Optional[List[str]] = None,
    timeout: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Upload multi-file code to E2B sandbox and execute.
    
    Args:
        code_str: Generated code with # FILE: markers
        main_file: Entry point filename (auto-detected if None)
        install_packages: List of packages to pip install
        timeout: Execution timeout in milliseconds
        verbose: Print status messages
        
    Returns:
        dict: {success, exit_code, stdout, stderr, files, error}
    """
    
    if verbose:
        print("\n" + "="*80)
        print("E2B SANDBOX EXECUTION")
        print("="*80 + "\n")
    
    try:
        # STEP 1: Parse code
        files = parse_multifile_code(code_str)
        
        if verbose:
            print(f"📁 PARSING: {len(files)} file(s)")
            for fname in files.keys():
                print(f"   - {fname} ({len(files[fname])} chars)")
            print()
        
        # STEP 2: Detect main file
        if main_file is None:
            main_file = detect_main_file(files)
        
        if main_file not in files:
            return {
                "success": False,
                "error": f"Main file '{main_file}' not found",
                "files": files,
                "exit_code": -1,
                "stdout": "",
                "stderr": ""
            }
        
        if verbose:
            print(f"🎯 ENTRY POINT: {main_file}\n")
        
        # STEP 3: Create sandbox
        if verbose:
            print(f"🚀 CREATING SANDBOX")
        
        sandbox = Sandbox.create(template="code-interpreter-v1")
        
        if verbose:
            print(f"   Sandbox ID: {sandbox.sandbox_id}\n")
        
        # STEP 4: Install packages
        if install_packages:
            if verbose:
                print(f"📦 INSTALLING PACKAGES")
            
            for package in install_packages:
                if verbose:
                    print(f"   Installing {package}...", end=" ")
                
                try:
                    if package.lower() in ["pytorch", "torch"]:
                        cmd = "pip install torch --index-url https://download.pytorch.org/whl/cpu"
                    elif package.lower() in ["tensorflow"]:
                        cmd = "pip install --no-cache-dir tensorflow-cpu"
                    elif package.lower() in ["opencv", "cv2"]:
                        cmd = "pip install opencv-python-headless"
                    elif package.lower() in ["os", "sys", "json", "re", "datetime", "time"]:
                        if verbose:
                            print("(built-in)")
                        continue
                    else:
                        cmd = f"pip install {package}"
                    
                    sandbox.commands.run(cmd, timeout=0) # cpu version of  torch, lighter (the sandbox is 1GB)

                    
                    if verbose:
                        print("✓")
                except Exception as e:
                    if verbose:
                        #print(f"✗ ({str(e)[:30]})")
                        print(f"✗ {str(e)}")
            
            if verbose:
                print()
        
        # STEP 5: Upload files
        if verbose:
            print(f"📤 UPLOADING FILES")
        
        upload_dir = "/home/user"
        sandbox.commands.run(f"mkdir -p {upload_dir}", timeout=0)
        
        for filename, content in files.items():
            remote_path = f"{upload_dir}/{filename}"
            
            try:
                sandbox.files.write(remote_path, content)
                if verbose:
                    print(f"   ✓ {filename}")
            except Exception as e:
                sandbox.kill()
                return {
                    "success": False,
                    "error": f"Failed to upload {filename}: {str(e)}",
                    "files": files,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": ""
                }
        
        if verbose:
            print()
        
        # STEP 6: Execute
        if verbose:
            print(f"▶️  EXECUTING {main_file}")
            print("="*80)
            print()
        
        # exec_result = sandbox.commands.run(
        #     f"cd {upload_dir} && python {main_file}",
        #     timeout=timeout
        # )

        exec_result = sandbox.commands.run(
            f"cd {upload_dir} && python {main_file}",
            timeout=timeout
        )


        
        if verbose:
            print("="*80)
            print()
            print(f"📊 RESULTS:")
            print(f"   Exit code: {exec_result.exit_code}")
            if exec_result.stdout:
                print(f"\n   STDOUT:\n{exec_result.stdout}")
            if exec_result.stderr:
                print(f"\n   STDERR:\n{exec_result.stderr}")
            print()
        
        # STEP 7: Cleanup
        sandbox.kill()
        
        return {
            "success": exec_result.exit_code == 0,
            "exit_code": exec_result.exit_code,
            "stdout": exec_result.stdout or "",
            "stderr": exec_result.stderr or "",
            "files": files,
            "error": None if exec_result.exit_code == 0 else "Non-zero exit"
        }
    
    except Exception as e:
        if 'sandbox' in locals():
            try:
                sandbox.kill()
            except:
                pass
        
        return {
            "success": False,
            "error": f"E2B execution failed: {str(e)}",
            "files": {},
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e)
        }


def check_code_sandbox(state: SummaryState, config: RunnableConfig):
    """
    Check code in E2B sandbox with multi-file support.
    """
    
    print("\n" + "="*80)
    print("SANDBOX EXECUTION WITH E2B")
    print("="*80 + "\n")
    
    # Get code from state
    code = state.normalized_code if state.user_feedback_processed == "evaluation" else state.code

    print("[SANDBOX] Converting relative imports to absolute...\n")
    code = code.replace('from .', 'from ')
    
    # Extract imports
    print("Extracting dependencies...\n")
    
    agent = Agent(
        model=Ollama(id="qwen3:latest"),
        tools=[],
        show_tool_calls=False,
    )
    
    imports_extractor = """You are given Python code. Return ONLY import statements.
Do NOT include explanations or code fences. Just imports."""
    
    response = agent.run(imports_extractor + "\n\nCode:\n" + code)
    imports = response.content
    
    # Extract packages
    try:
        packages = extract_packages_from_imports(imports)
        print(f"Packages to install: {packages}\n")
    except Exception as e:
        print(f"Package extraction failed: {e}\n")
        packages = []
    
    # Static type check
    print("Running static type check...\n")
    static_evaluator = create_pyright_evaluator()
    static_result = static_evaluator(outputs=code)
    
    # Clean code
    cleaned_code = code.replace("```python\n", "").replace("\n```", "")
    
    # Execute in E2B
    print("Executing in E2B sandbox...\n")
    
    sandbox_result = upload_and_execute_in_e2b(
        code_str=cleaned_code,
        main_file=None,
        install_packages=packages,
        timeout=0,
        verbose=True
    )
    
    # Store results
    state.sandbox_execution_result = sandbox_result
    
    # Prepare feedback
    if sandbox_result["success"]:
        feedback = f"✅ Execution successful.\n\nOutput:\n{sandbox_result['stdout']}"
    else:
        error_msg = sandbox_result.get("stderr", sandbox_result.get("error", "Unknown error"))
        feedback = f"❌ Execution failed.\n\nError:\n{error_msg}"
    
    return {
        #"sandbox_feedback_pyright": static_result or "No result",
        "sandbox_feedback_execution": {
            "status": "executed",
            "exit_code": sandbox_result["exit_code"],
            "success": sandbox_result["success"]
        },
        "sandbox_execution_result": sandbox_result,
        "sandbox_feedback_user": feedback
    }





# def check_code_sandbox(state: SummaryState, config: RunnableConfig):
#     """
#     Check code in a sandbox with dependency installation.
#     """
#     print("---CHECKING CODE IN SANDBOX---")
#     #agent that extracts ONLY the imports

#     code = ""

#     #print("state in check_code_sandbox: ", state)

#     print("Current state.user_feedback_processed: ", state.user_feedback_processed)
#     if state.user_feedback_processed == "evaluation":
#         code = state.normalized_code # from normalization step
#     elif state.user_feedback_processed == "execute":
#         code = state.code # from code generation step

#     agent = Agent(
#         model=Ollama(id="qwen3:latest"),
#         tools=[],
#         show_tool_calls=False,
#         use_json_mode=False,
#     )
    
#     imports_extractor_instructions = """
#     You are given a block of Python code.
#     Your task is to return ONLY the import statements as a single block of text.
#     ### Instructions:
#     - Output ONLY the import statements, preserving their order, Separated by new lines.
#     - Do NOT include explanations, markdown, quotes, or code fences.
#     - Exclude any comments or non-import lines.

#     - example:
#         import torch
#         import snntorch as snn
        
#     """
#     #print("Code to extract imports from:\n", code)
#     query = imports_extractor_instructions + "\n Code:\n" + code + "\n"

#     response = agent.run(query)

    
#     imports = response.content
    
#     print("Imports to give to extract_packages:", imports)
#     #code = state.code_generation.code or ""
    
#     cleaned_code = code.replace("```python\n", "").replace("\n```", "") # remove markdown formatting if any
#     #combined_code = f"{imports}\n{code}"
    


#     #sandbox_pyright = Sandbox.create("OpenEvalsPython") # already with pyright and uv installed
#     static_evaluator = create_pyright_evaluator()


#     sandbox_execution = Sandbox.create(
#             template="code-interpreter-v1",
#     ) # with this use sbx.run_code
    
#     metrics = sandbox_execution.get_metrics() 
#     print("Sandbox metrics:", metrics)
    
#     # Static type check
#     static_evaluation_result = static_evaluator(outputs=cleaned_code)
    
#     # Extract dependencies via LLM
#     try:
#         packages = extract_packages_from_imports(imports)
#         print("Inferred packages:", packages)
#     except Exception as e:
#         print("Fallback to empty package list due to error:", e)
#         packages = []

#     # Install packages
#     for pkg in packages:
#         print(f"Installing {pkg} in sandbox...")

#         if pkg in ["pytorch","torch"]:
#             sandbox_execution.commands.run("pip install torch --index-url https://download.pytorch.org/whl/cpu", timeout=0) # cpu version of  torch, lighter (the sandbox is 1GB)
#         elif pkg in ["tensorflow"]:
#             sandbox_execution.commands.run("pip install --no-cache-dir tensorflow-cpu", timeout=0) # cpu version of tensorflow, lighter
#         elif pkg in ["sklearn"]:
#             sandbox_execution.commands.run("pip install scikit-learn", timeout=0) # sklearn is actually scikit-learn
#         elif pkg in ["opencv", "cv2"]:
#             sandbox_execution.commands.run("pip install opencv-python-headless", timeout=0)
#         elif pkg in ["PIL"]:
#             sandbox_execution.commands.run("pip install Pillow", timeout=0)
#         elif pkg in ["torchvision"]:
#             sandbox_execution.commands.run("pip install --no-deps --only-binary=:all: torchvision", timeout=0)
#         elif pkg in ["os"]:
#             print("Skipping installation of built-in package 'os'")
#         else:
#             result = sandbox_execution.commands.run(f"pip install {pkg}", timeout=0)
#             print(result.stdout or result.stderr)

#     print("code to execute in sandbox:\n", cleaned_code)

#     # Execution check
#     evaluator_exec = create_e2b_execution_evaluator(sandbox=sandbox_execution)
#     eval_result_execution = evaluator_exec(outputs=cleaned_code)
#     print("Execution result:", eval_result_execution)

#     # Direct execution in sandbox (outside evaluator)
#     run_result = sandbox_execution.run_code(cleaned_code)

#     sandbox_execution.kill()

#     print("=== STDOUT ===")
#     print(run_result)
    
#     state.sandbox_execution_result = run_result
    

#     return {
#         "sandbox_feedback_pyright": static_evaluation_result or "No pyright result",
#         "sandbox_feedback_execution": eval_result_execution,
#         "sandbox_execution_result": run_result
#     }





def reflection(state: SummaryState, config: RunnableConfig):
    """
    Reflect on the code and decide whether to finish or retry.
    """

    print("---REFLECTING ON CODE---")

    # Check code
    pyright_feedback = state.sandbox_feedback_pyright
    execution_feedback = state.sandbox_feedback_execution
    print("PYRIGHT FEEDBACK:", pyright_feedback)

    
    agent = Agent(
        model=Ollama(id="mistral:latest"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    # Convert dictionaries to strings
    pyright_feedback_str = json.dumps(pyright_feedback, indent=2)
    execution_feedback_str = json.dumps(execution_feedback, indent=2)

    # Construct the query using f-strings for better readability
    query = code_reflection_instructions + "\n"  + "\n" + execution_feedback_str + pyright_feedback_str
    response = agent.run(query)

    # Get the string content from RunResponse
    response_content = response.content if hasattr(response, 'content') else str(response)
    print("REFLECTION RESPONSE:", response_content)


    # Store response in a variable
    code_reflection = response_content

    code_reflection = "Few checks on the generated code" + code_reflection

    # Reset research topic for next iteration
    state.research_topic = ""

    return {
        "code_reflection": code_reflection,
        
    }


### NOT USED
def decide_to_finish(state: SummaryState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state.error
    

    if error == "no" or state.code_iterations == state.max_code_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


# def collect_feedback(state: SummaryState, config: RunnableConfig):
#     print("---COLLECTING FEEDBACK FROM USER---")

#     feedback_prompt = {
#         "instruction": "Please review the code and choose: approve, regenerate. Then explain your decision.",
#         "code_review": {
#             "prefix": state.code_generation.prefix,
#             "imports": state.code_generation.imports,
#             "code": state.code_generation.code,
#         }
#     }

#     feed = interrupt(feedback_prompt)
    
#     print("---FEEDBACK COLLECTED--", feed)
#     print("\n")

def collect_feedback(state: SummaryState, config: RunnableConfig):
    print("---COLLECTING FEEDBACK FROM USER---")

    #sandbox_pyright = Sandbox.create("OpenEvalsPython") # already with pyright and uv installed
    static_evaluator = create_pyright_evaluator()

    code = state.code


    # ensure 'code' is a dict
    if isinstance(code, dict):
        code = code.get("code", "")

    code = code.replace("```python\n", "").replace("\n```", "")
    #print("Combined code for static evaluation:\n", combined_code)
    

    static_evaluation_result = static_evaluator(outputs=code)

    

    feedback_prompt = {
        "instruction": "Please review the code and choose: *approve*, *regenerate*, *evaluate with a reference code* or to *execute the code*. Eventually explain your decision.",
        "static_python_check": static_evaluation_result or "No pyright result",
    }

    # This pauses execution and will show editable fields
    user_input = interrupt(feedback_prompt)

    # Return the user edits into state
    # return {
    #     "user_feedback": user_input
    # }







# def process_feedback(state: SummaryState, config: RunnableConfig):
#     """Process the user feedback and classify next action."""
#     print("---PROCESSING FEEDBACK DECISION---")

#     #print("\nstate in process feedback:", state)

#     agent = Agent(
#         model=Ollama(id="deepseek-r1"),
#         tools=[],
#         show_tool_calls=False,
#         structured_outputs=True,
#     )

#     prompt = f"""\

#     You are a decision agent.

#     You are given user feedback: {state.research_topic}

#     Based on this feedback, respond with **only one** of the following exact JSON objects:

#     {{"response": "approve"}}
#     {{"response": "regenerate"}}
#     {{"response": "evaluation"}}
#     {{"response": "execute"}}

#     ### Definitions:
#     - Use **"evaluation"** if the user wants to perform an evaluation or if the user talks about to evaluate.
#     - Use **"approve"** if the user is fully satisfied and wants to keep the code exactly as it is, with **no changes requested**.
#     - Use **"regenerate"** if the user asks for **any modifications**, **improvements**, or expresses **dissatisfaction** with the current code.
#     - Use **"execute"** if the user wants to run the code in a sandbox to check for errors.
#     Only return the JSON object — do not include any other text, explanations, or logs.
#     """


    


#     try:
#         response = agent.run(prompt)
#         response_text = getattr(response, "content", str(response))
#         print(f"RAW DECISION RESPONSE: {response_text}")

#         response_text = remove_think_tags(response_text)

#         print(f"RESPONSE TEXT: {response_text}")

#         parsed = json.loads(response_text)
#         action = parsed.get("response", "regenerate").lower()

#         if action not in {"regenerate", "approve", "evaluation", "execute"}:
#             action = "regenerate"

#     except Exception as e:
#         print(f"Error parsing feedback response: {e}")
#         action = "regenerate"

#     state.user_feedback_processed = action
#     return {"user_feedback_processed": action}

def process_feedback(state: SummaryState, config: RunnableConfig):
    """Process the user feedback and classify next action with optional filename."""
    print("---PROCESSING FEEDBACK DECISION---")

    agent = Agent(
        model=Ollama(id="deepseek-r1"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,  
    )

    prompt = f"""\
You are a decision agent that classifies user feedback into actions.

User feedback: {state.research_topic}

Respond with ONLY ONE JSON object (no markdown, no explanations):

For approve/regenerate/evaluation:
{{"response": "approve"}}
{{"response": "regenerate"}}
{{"response": "evaluation"}}

For execute (with optional filename):
{{"response": "execute"}}
{{"response": "execute", "file_name": "train.py"}}
{{"response": "execute", "file_name": "main.py"}}

### Definitions:
- "evaluation": if user wants to evaluate the code
- "approve": if user is fully satisfied, wants no changes
- "regenerate": if user asks for modifications or improvements
- "execute": if user wants to run/test the code in sandbox
  - Include "file_name" ONLY if user specifies which file (e.g., "execute train.py")
  - Omit "file_name" if user just says "run it" or "execute"

### Examples:
User: "execute train.py" → {{"response": "execute", "file_name": "train.py"}}
User: "run main.py" → {{"response": "execute", "file_name": "main.py"}}
User: "test config.py" → {{"response": "execute", "file_name": "config.py"}}
User: "execute" → {{"response": "execute"}}
User: "run it" → {{"response": "execute"}}
User: "Make it faster" → {{"response": "regenerate"}}
User: "Looks good" → {{"response": "approve"}}
User: "Please evaluate this" → {{"response": "evaluation"}}

Return ONLY valid JSON. No markdown. No extra text.
"""

    response = agent.run(prompt)

    try:
        # Use YOUR function here!
        content = remove_think_tags(response.content).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            response_dict = json.loads(json_match.group())
        else:
            response_dict = json.loads(content)

        decision = response_dict.get('response', 'approve')
        file_name = response_dict.get('file_name', None)

        print(f"✓ Decision: {decision}")
        if file_name:
            print(f"✓ File to execute: {file_name}")

        state.user_feedback_processed = decision
        state.user_specified_file = file_name

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"⚠️ Failed to parse response: {e}")
        state.user_feedback_processed = 'approve'
        state.user_specified_file = None

    return state

def collect_feedback_evaluation(state: SummaryState, config: RunnableConfig):
    print("---COLLECTING FEEDBACK FOR EVALUATION FROM USER---")

    feedback_prompt = {
        "instruction": "Please insert the *REFERENCE CODE* to compare the generated code with.",
        "code_review": {
            "code": state.code,
        }
    }

    interrupt(feedback_prompt)





def code_normalization(state: SummaryState, config: RunnableConfig):
    """Process the user feedback, produce the codes to compare with the same inputs"""

    print("---NORMALIZING GENERATED CODE AND REFERENCE CODE---")

    agent = Agent(
        model=Ollama(id="gpt-oss:20b"),#deepseek-r1
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    
    script_str = ""

    code_normalization_query = code_normalization_instructions.format(code=state.code, research_topic=state.research_topic)

    try:
        response = agent.run(code_normalization_query)
        script_str = getattr(response, "content", str(response))

        

    except Exception as e:
        print(f"An error occurred: {e}")

    print("NORMALIZED CODE:\n", script_str)
    state.normalized_code = script_str
    return {"normalized_code": script_str}

def collect_feedback_normalization(state: SummaryState, config: RunnableConfig):
    print("---COLLECTING FEEDBACK FOR NORMALIZATION FROM USER---")
    feedback_prompt = {
        "instruction": "Please review the normalized code and insert any corrections if needed.",
        "normalized_code": state.normalized_code
    }
    interrupt(feedback_prompt)




def process_feedback_normalization(state: SummaryState, config: RunnableConfig):
    """
    Process the user feedback for normalization.
    Ensures the returned result is always valid JSON with a 'fixed_code' field
    containing a Python code block.
    """
    print("---PROCESSING FEEDBACK DECISION FOR NORMALIZATION---")

    agent = Agent(
        model=Ollama(id="gpt-oss:20b"),#deepseek-r1
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    prompt = f"""
    You are a code fixer.

    You are given user feedback: {state.research_topic}

    Based on this feedback, modify the normalization code as needed.

    The code to modify is:
    {state.normalized_code}

    Return ONLY a JSON object with a single key "fixed_code".

    """

    try:
        response = agent.run(prompt)
        response_text = getattr(response, "content", str(response))

        

        response_text = remove_think_tags(response_text)

        # Strip markdown code fences
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines() if not line.startswith("```")
            ).strip()

        fixed_code = ""
        try:
            # Try to parse as JSON
            parsed = json.loads(cleaned)
            fixed_code = parsed.get("fixed_code", "")
        except json.JSONDecodeError:
            # If it's raw code, wrap it into valid JSON with Python code block
            fixed_code = f"```python\n{cleaned}\n```"
            cleaned = json.dumps({"fixed_code": fixed_code})
            parsed = json.loads(cleaned)
            fixed_code = parsed["fixed_code"]

        print(f"FIXED CODE:\n{fixed_code}")  # Display with code block

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"normalized_code": ""}

    #state.fixed_code = fixed_code
    #state.fixed_code = f"```python\n{fixed_code}\n```"
    state.normalized_code = fixed_code

    return {"normalized_code": fixed_code}


def add_performance_metrics(state: SummaryState, config: RunnableConfig):
    """
    Add performance quality metrics to the code.
    """
    print("---ADDING METRICS TO CODE---")

    agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    


    prompt = """
        Task: Write a complete Python script that defines a reusable function to measure and report performance metrics for any PyTorch or snnTorch model on a single forward pass. The function should accept a model instance and an input tensor, then print:

        Model: <class name>
        Forward pass time: <seconds> s
        Total parameters: <count>
        Peak memory usage: <MB>

        Rules:
        - Do not modify model architectures or forward computations.
        - Do not include explanations, markdown, comments outside the code, or special characters like ### or <br>.
        - Output ONLY the complete Python script.
        - Handle models with no trainable parameters gracefully.
        - Handle models without .parameters() gracefully.
        - Default to torch.device("cpu") if device is not available.
        - Use safe device detection logic.
        - Use time.time() for timing.
        - Use torch.cuda.reset_peak_memory_stats() / torch.cuda.max_memory_allocated() for CUDA memory.
        - Use tracemalloc for CPU memory tracking if CUDA is not available.
        - Count parameters with sum(p.numel() for p in model.parameters()) if available, else return 0.
        - After execution, demonstrate the function on two models: one "reference model" and one "generated model" (both can be simple PyTorch nn.Module examples).

    """




    query= prompt + "\n" + state.normalized_code
    run_response = agent.run(query)
    #print("RAW SEARCH RESPONSE:\n", run_response)
    content = run_response.content
    print(f"Code with metrics added:\n{content}")
    state.normalized_code = content
    return {"normalized_code": content}





####################################### GRAPH BUILDING #########################################
# Add nodes and edges 


builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)


# node for routing
builder.add_node("route_question", route_question)

# nodes for web research
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)
builder.add_node("json_parser", json_parser)

# node for academic research
builder.add_node("academic_research", academic_research)
builder.add_node("summarize_academic_sources", summarize_academic_sources)
builder.add_node("generate_academic_query", generate_academic_query)
builder.add_node("finalize_academic_summary", finalize_academic_summary)
builder.add_node("reflect_on_academic_summary", reflect_on_academic_summary)



# Add nodes for code generation and code checking
builder.add_node("search_relevant_sources", search_relevant_sources)

builder.add_node("generate", generate)  # generation solution
builder.add_node("collect_feedback", collect_feedback)
builder.add_node("process_feedback", process_feedback)
builder.add_node("check_code_sandbox", check_code_sandbox)  # check code in sandbox
builder.add_node("reflection", reflection)  # reflect on code errors

builder.add_node("collect_feedback_evaluation", collect_feedback_evaluation)  # collect feedback for evaluation
builder.add_node("code_normalization", code_normalization)  # process feedback for evaluation

builder.add_node("collect_feedback_normalization", collect_feedback_normalization) # collect feedback for normalization
builder.add_node("process_feedback_normalization", process_feedback_normalization) # process feedback for normalization

builder.add_node("add_performance_metrics", add_performance_metrics)  # add performance metrics to code

########################## EGDES ##########################
builder.add_edge(START, "route_question")


builder.add_edge("generate_query", "web_research")
builder.add_conditional_edges(
    "route_question",
    lambda state: state.route,  # Read route from state
    {
        "Academic Source": "generate_academic_query",
        "General Web Search": "generate_query",
        "Code": "search_relevant_sources",
    }
)

builder.add_edge("web_research", "json_parser")
builder.add_edge("json_parser", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)

builder.add_edge("finalize_summary", END)

builder.add_edge("generate_academic_query", "academic_research")
#builder.add_edge("academic_research", "summarize_academic_sources")
#builder.add_edge("summarize_academic_sources", END)
#builder.add_edge("summarize_academic_sources", "finalize_academic_summary")
builder.add_edge("summarize_academic_sources", "reflect_on_academic_summary")

# From reflection, route based on loop count:
builder.add_conditional_edges(
    "reflect_on_academic_summary",
    route_academic_research,
    {
        "academic_research": "academic_research",
        "finalize_academic_summary": "finalize_academic_summary"
    }
)

# Back to research if continuing:
builder.add_edge("academic_research", "summarize_academic_sources")

# Finalize to END (already exists):
builder.add_edge("finalize_academic_summary", END)

builder.add_edge("search_relevant_sources", "generate")

builder.add_edge("generate", "collect_feedback")

builder.add_edge("check_code_sandbox", "reflection")

builder.add_edge("reflection", "collect_feedback")


builder.add_edge("collect_feedback", "process_feedback")

builder.add_conditional_edges(
    "process_feedback",
    lambda state: state.user_feedback_processed,
    {
        "regenerate": "generate",
        "approve": END,
        "evaluation": "collect_feedback_evaluation",
        "execute": "check_code_sandbox",
    },
)

builder.add_edge("collect_feedback_evaluation", "code_normalization")
builder.add_edge("code_normalization", "collect_feedback_normalization")
builder.add_edge("collect_feedback_normalization", "process_feedback_normalization")
builder.add_edge("process_feedback_normalization", "add_performance_metrics")
builder.add_edge("add_performance_metrics", "check_code_sandbox")



# Add memory saver for checkpointing
memory = MemorySaver()

# Compile graph with checkpointing and interrupts
graph = builder.compile(
#checkpointer=memory,
#interrupt_after=[""],
)

