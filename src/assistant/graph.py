#langgraph dev --no-reload --allow-blocking  # to run without reloading and allow blocking operations (to use DeepEval)

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


# -------------------------
# CopilotKit / LangGraph Utilities
# -------------------------
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_exit, copilotkit_customize_config


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



############################################### WEB RESEARCH BRANCH ###############################################

async def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """
    
    # Format the prompt
    #print(f"Current state: {state}")
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = await llm_json_mode.ainvoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )   
    query = json.loads(result.content)
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Generate Query",
        "content": query['query']
    }))
    return {"search_query": query['query']}


async def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    print("IN WEB RESEARCH")    
    max_results = 5
    include_raw_content = True

    # Customize config to not emit tool calls during web search
    config = copilotkit_customize_config(config, emit_tool_calls=False)
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Web Research",
        "content": f"Searching for: {state.search_query}"
    }))
    
    # Search the web
    agent = Agent(
        model=Ollama(id="qwen3:latest"),
        tools=[TavilyTools()], #GoogleSearchTools() DuckDuckGoTools()
        show_tool_calls=False,
        markdown=True
        
    )

    query = web_search_instructions + "\n Search for the following query: " + state.search_query + "\n"
    #run_response = agent.run(query)
    #run_response = await agent.arun(query)  # ✅ Non-blocking!
    run_response = await asyncio.to_thread(lambda: agent.run(query))

    print("RAW SEARCH RESPONSE:\n", run_response)

    content = run_response.content

    return {"raw_search_result": content}







async def json_parser(state: SummaryState, config: RunnableConfig):
    """ Parse the JSON output from the web search """
    await copilotkit_emit_message(config, json.dumps({
        "node": "JSON Pasrser",
        "content": "Parsing JSON output..."
    }))

    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=configurable.local_llm, temperature=0)
    result = await llm.ainvoke(
        [SystemMessage(content=web_search_expected_output),
        HumanMessage(content=state.raw_search_result)]
    )
    content = result.content
    start_index = content.find("{")
    strip_content = content[start_index:] if start_index != -1 else content
    end_index = strip_content.rfind("}")
    search_str = strip_content[:end_index+1] if end_index != -1 else strip_content

    [print(f"Search results: {search_str}")]

    search_results = json.loads(search_str)
    # Format the sources
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=3000)
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Web Research",
        "content": "Found and processed search results"
    }))
    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

async def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Sources",
        "content": "Summarizing gathered information..."
    }))
    
    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=configurable.local_llm, temperature=0)
    result = await llm.ainvoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content



    running_summary = remove_think_tags(running_summary)

    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Sources",
        "content": running_summary
    }))
    return {"running_summary": running_summary}

async def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    await copilotkit_emit_message(config, json.dumps({
        "node": "Reflect on Summary",
        "content": "Analyzing current findings for gaps in knowledge..."
    }))
    
    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = await llm_json_mode.ainvoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )   
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

# async def finalize_summary(state: SummaryState, config: RunnableConfig):
#     """ Finalize the summary """
    
#     await copilotkit_emit_message(config, json.dumps({
#         "node": "Finalize Summary",
#         "content": "Finalizing research summary..."
#     }))
    
#     # Format all accumulated sources into a single bulleted list
#     all_sources = "\n".join(source for source in state.sources_gathered)
#     final_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    
#     await copilotkit_emit_message(config, json.dumps({
#         "node": "Finalize Summary",
#         "content": final_summary
#     }))
    
#     # Signal completion to copilotkit
#     await copilotkit_exit(config)
    
#     return {"running_summary": final_summary}

import asyncio

# ===== STEP 1: NEW FUNCTION - Synchronous evaluation (runs in thread) =====
def _evaluate_summary_sync(
    research_topic: str,
    running_summary: str,
    web_research_results: list
) -> dict:
    """
    SYNCHRONOUS evaluation - runs in separate thread.
    Uses metrics compatible with web search.
    """
    print("\n🔍 Running DeepEval evaluation with Ollama (deepseek-r1:latest)...\n")
    
    try:
        # ===== CRITICAL: SET ENVIRONMENT VARIABLES FIRST =====
        import os
        os.environ["DEEPEVAL_RESULTS_FOLDER"] = ""
        os.environ["DEEPEVAL_DISABLE_TELEMETRY"] = "1"
        # Force no caching
        os.environ["DEEPEVAL_SKIP_PROMPTS_CACHE"] = "1"
        # ===== THEN import DeepEval AFTER setting env vars =====
        
        from deepeval import evaluate
        from deepeval.models import OllamaModel
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            HallucinationMetric
        )
        from deepeval.test_case import LLMTestCase
        
        # Initialize Ollama model
        ollama_model = OllamaModel(
            model="deepseek-r1:latest",
            base_url="http://localhost:11434"
        )
        
        # Define metrics that work with web search
        faithfulness = FaithfulnessMetric(threshold=0.55, model=ollama_model) 
        relevancy = AnswerRelevancyMetric(threshold=0.55, model=ollama_model)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.55, model=ollama_model)
        hallucination = HallucinationMetric(threshold=0.55, model=ollama_model)
        
        # Create test case with BOTH context AND retrieval_context
        # FaithfulnessMetric requires BOTH
        test_case = LLMTestCase(
            input=research_topic,
            actual_output=running_summary,
            context=web_research_results,              # ← For Hallucination & ContextualRelevancy
            retrieval_context=web_research_results     # ← For Faithfulness
        )
        
        # Run evaluation with 4 compatible metrics
        result = evaluate(
            [test_case],
            [
                faithfulness,
                relevancy,
                contextual_relevancy,
                hallucination
            ]
        )
        
        
        
        # Return success with all metric scores
        return {
            "completed": True,
            "status": "EVALUATED",
            "total_metrics": 4
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "completed": False,
            "error": str(e),
            "status": "ERROR"
        }


# ===== STEP 2: MODIFIED evaluate_summary_with_interrupt =====
async def evaluate_summary_with_interrupt(
    research_topic: str,
    running_summary: str,
    web_research_results: list
) -> dict:
    """
    Evaluate the summary with user interrupt.
    Runs DeepEval in separate thread (bypasses uvloop).
    Uses 4 web search-compatible metrics.
    """
    
    print("\n" + "="*80)
    print("GENERATED SUMMARY:")
    print("="*80)
    print(running_summary)
    print("="*80 + "\n")

    user_input = "yes"
    
    # Get user input in thread (non-blocking)
    #user_input = await asyncio.to_thread(
    #    input,
    #    "Do you want to evaluate this summary? (yes/no): "
    #)

    user_input = user_input.strip().lower()
        
    if user_input in ["yes", "y"]:
        # ===== CRITICAL: Run evaluation in SEPARATE THREAD =====
        evaluation_result = await asyncio.to_thread(
            _evaluate_summary_sync,
            research_topic,
            running_summary,
            web_research_results
        )
        # ===== END THREAD EXECUTION =====
        
        return evaluation_result
    
    else:
        print("\n⏭️  No evaluation performed. Continuing...\n")
        return {
            "completed": False,
            "status": "SKIPPED"
        }


# ===== STEP 3: MODIFIED finalize_summary =====
async def finalize_summary(state, config):
    """
    Modified finalize_summary that includes comprehensive evaluation.
    Uses 4 RAG metrics compatible with web search:
    Faithfulness, Answer Relevancy, Contextual Relevancy, Hallucination.
    """
    
    # Your existing finalize logic
    all_sources = "\n".join(source for source in state.sources_gathered)
    final_summary = f"## Summary\n\n{state.running_summary}\n\n### Sources:\n{all_sources}"
    
    # ===== COMPREHENSIVE EVALUATION (using thread-based approach) =====
    evaluation_result = await evaluate_summary_with_interrupt(
        research_topic=state.research_topic,
        running_summary=state.running_summary,
        web_research_results=state.web_research_results
    )
    # ===== END EVALUATION =====
    
    await copilotkit_exit(config)
    
    return {
        "running_summary": final_summary,
        "evaluation_results": evaluation_result
    }





def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary" 
   


######################################## ACADEMIC RESEARCH BRANCH ############################################



async def generate_academic_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for academic search """
    
    # Format the prompt
    #print(f"Current state: {state}")
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = await llm_json_mode.ainvoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for an academic search:")]
    )   
    query = json.loads(result.content)
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Generate Query",
        "content": query['query']
    }))
    return {"search_query": query['query']}






# async def academic_research(state, config, papers_limit=3):
#     query = state.get("search_query") if isinstance(state, dict) else getattr(state, "search_query", "")

#     # scholarly is synchronous, so run in a thread to avoid blocking event loop
#     def fetch_papers():
#         search_query = scholarly.search_pubs(query)
#         results = []
#         for _ in range(papers_limit):
#             try:
#                 paper = next(search_query)
#                 title = paper.get('bib', {}).get('title', 'No title')
#                 abstract = paper.get('bib', {}).get('abstract', 'No abstract')
#                 results.append(f"{title} - {abstract}")
#             except StopIteration:
#                 break
#         return results

#     abstracts = await asyncio.to_thread(fetch_papers)

#     return {"academic_source_content": abstracts}


async def academic_research(state, config, papers_limit=3):
    query = state.get("search_query") if isinstance(state, dict) else getattr(state, "search_query", "")

    agent = Agent(
        model=Ollama(id="qwen3:latest"),
        tools=[ArxivTools()],
        show_tool_calls=False,
        markdown=True
        
    )

    academic_query = f"Search for academic papers on the following topic: {query}\n"
    run_response = await asyncio.to_thread(lambda: agent.run(academic_query))
    print("RAW ACADEMIC SEARCH RESPONSE:\n", run_response)
    content = run_response.content
    return {"academic_source_content": content, "research_loop_count": state.research_loop_count + 1}



    


async def summarize_academic_sources(state: SummaryState, config: RunnableConfig):

    print("IN SUMMARIZE ACADEMIC RESEARCH")
    """ Summarize the gathered sources from academic research """
    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Academic Sources",
        "content": "Summarizing gathered information..."
    }))
    
    
    
    # Existing summary
    content = state.academic_source_content
    


    # Build the human message
    if content:
        human_message_content = (
            f"Produce the summary of: {content}\n\n"
        )
    else:
        human_message_content = "empty"
    

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=configurable.local_llm, temperature=0)
    result = await llm.ainvoke(
        [SystemMessage(content=academic_summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content


    running_summary = remove_think_tags(running_summary)

    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Sources",
        "content": running_summary
    }))
    return {"running_summary": running_summary}




async def reflect_on_academic_summary(state: SummaryState, config: RunnableConfig):
    """
    Reflect on the academic summary and generate a follow-up query.
    EXACT SAME APPROACH as reflect_on_summary but for academic sources.
    """
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Reflect on Academic Summary",
        "content": "Analyzing current academic findings for gaps in knowledge..."
    }))
    
    configurable = Configuration.from_runnable_config(config)
    
    # EXACT SAME LLM setup as web search reflection
    llm_json_mode = ChatOllama(
        model=configurable.local_llm,
        temperature=0,
        format="json"
    )
    
    result = await llm_json_mode.ainvoke(
        [
            SystemMessage(content=reflection_instructions.format(
                research_topic=state.research_topic
            )),
            HumanMessage(content=f"Identify a knowledge gap and generate a follow-up academic search query based on our existing knowledge\n\n{state.running_summary}")
        ]
    )
    
    followup_query = json.loads(result.content)
    query = followup_query.get("followup_query")
    
    if not query:
        return {"search_query": f"Tell me more about {state.research_topic}"}
    
    return {"search_query": followup_query}


# ===== ROUTE FUNCTION FOR ACADEMIC =====
# Same routing logic as route_research

async def route_academic_research(state: SummaryState, config: RunnableConfig):
    """
    Route academic research based on loop count.
    EXACT SAME as route_research but for academic branch.
    """
    
    # Check research loop count
    if state.research_loop_count >= 2:  # max 2 loops for academic
        return "finalize_academic_summary"
    else:
        return "academic_research"  # Continue researching



def _evaluate_academic_summary_sync(
    research_topic: str,
    running_summary: str,
    academic_sources: list
) -> dict:
    """
    SYNCHRONOUS evaluation for academic search - runs in separate thread.
    Uses same 4 metrics as web search for consistency.
    """
    print("\n🔍 Running DeepEval evaluation for Academic Search with Ollama (deepseek-r1:latest)...\n")
    
    try:
        # ===== CRITICAL: SET ENVIRONMENT VARIABLES FIRST =====
        import os
        os.environ["DEEPEVAL_RESULTS_FOLDER"] = ""
        os.environ["DEEPEVAL_DISABLE_TELEMETRY"] = "1"
        # Force no caching
        os.environ["DEEPEVAL_SKIP_PROMPTS_CACHE"] = "1"
        # ===== THEN import DeepEval AFTER setting env vars =====
        
        from deepeval import evaluate
        from deepeval.models import OllamaModel
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            HallucinationMetric
        )
        from deepeval.test_case import LLMTestCase
        
        # Initialize Ollama model
        ollama_model = OllamaModel(
            model="deepseek-r1:latest",
            base_url="http://localhost:11434"
        )
        
        # Define metrics that work with academic search
        faithfulness = FaithfulnessMetric(threshold=0.55, model=ollama_model)
        relevancy = AnswerRelevancyMetric(threshold=0.55, model=ollama_model)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.55, model=ollama_model)
        hallucination = HallucinationMetric(threshold=0.55, model=ollama_model)
        
        # Create test case with BOTH context AND retrieval_context
        test_case = LLMTestCase(
            input=research_topic,
            actual_output=running_summary,
            context=academic_sources,              # ← For Hallucination & ContextualRelevancy
            retrieval_context=academic_sources     # ← For Faithfulness
        )
        
        # Run evaluation with 4 compatible metrics
        result = evaluate(
            [test_case],
            [
                faithfulness,
                relevancy,
                contextual_relevancy,
                hallucination
            ]
        )
        
        # Extract scores after evaluation completes
        faithfulness_score = faithfulness.score if faithfulness.score is not None else 0.0
        relevancy_score = relevancy.score if relevancy.score is not None else 0.0
        contextual_relevancy_score = contextual_relevancy.score if contextual_relevancy.score is not None else 0.0
        hallucination_score = hallucination.score if hallucination.score is not None else 0.0
        
        
        
        # Return success with all metric scores
        return {
            "completed": True,
            "status": "EVALUATED",
            "total_metrics": 4,
            "source_type": "academic"
        }
        
    except Exception as e:
        print(f"\n❌ Academic Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "completed": False,
            "error": str(e),
            "status": "ERROR"
        }


async def evaluate_academic_summary_with_interrupt(
    research_topic: str,
    running_summary: str,
    academic_sources: list
) -> dict:
    """
    Evaluate the academic summary with user interrupt.
    Runs DeepEval in separate thread (bypasses uvloop).
    Uses same 4 metrics as web search for consistency.
    """
    
    print("\n" + "="*80)
    print("GENERATED ACADEMIC SUMMARY:")
    print("="*80)
    print(running_summary)
    print("="*80 + "\n")

    user_input = "yes"
    
    # Get user input in thread (non-blocking)
    #user_input = await asyncio.to_thread(
    #    input,
    #    "Do you want to evaluate this academic summary? (yes/no): "
    #)

    user_input = user_input.strip().lower()
        
    if user_input in ["yes", "y"]:
        # ===== CRITICAL: Run evaluation in SEPARATE THREAD =====
        evaluation_result = await asyncio.to_thread(
            _evaluate_academic_summary_sync,
            research_topic,
            running_summary,
            academic_sources
        )
        # ===== END THREAD EXECUTION =====
        
        return evaluation_result
    
    else:
        print("\n⏭️  No academic evaluation performed. Continuing...\n")
        return {
            "completed": False,
            "status": "SKIPPED"
        }


# ===== MODIFIED finalize_academic_summary =====
async def finalize_academic_summary(state, config):
    """
    Modified finalize_academic_summary that includes comprehensive evaluation.
    Uses same 4 RAG metrics as web search for consistency:
    Faithfulness, Answer Relevancy, Contextual Relevancy, Hallucination.
    """
    
    # Your existing finalize logic
    all_sources = state.academic_source_content if state.academic_source_content else "No academic sources"
    final_summary = f"## Academic Summary\n\n{state.running_summary}\n\n### Sources:\n{all_sources}"
    
    # ===== COMPREHENSIVE EVALUATION (using thread-based approach) =====
    evaluation_result = await evaluate_academic_summary_with_interrupt(
        research_topic=state.research_topic,
        running_summary=state.running_summary,
        academic_sources=[state.academic_source_content] if state.academic_source_content else []
    )
    # ===== END EVALUATION =====
    
    await copilotkit_exit(config)
    
    return {
        "running_summary": final_summary,
        "evaluation_results": evaluation_result
    }





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


def generate_optimized_search_queries(
    user_question: str,
    num_queries: int = 10,
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
    

    # --- Helper: Safe JSON extraction ---
    def safe_llm_json(content: str):
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
        return json.loads(content)

    query_generation_prompt = f"""You are an expert at generating comprehensive search queries for machine learning documentation retrieval.

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



import json
from typing import List

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

Generate 10 diverse, focused search queries covering architecture, parameters, training, modules, and usage patterns.

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

    # STEP 5: Extract patterns for prompt conditioning
    patterns = {
        "classes": list(set(re.findall(r"class\s+(\w+)\s*[:\(]", snn_context))),
        "functions": list(set(re.findall(r"def\s+(\w+)\s*\(", snn_context)))[:10],
        "modules": list(set(re.findall(r"(?:import|from)\s+([\w\.]+)", snn_context))),
    }

    print(f"   ✓ Classes: {len(patterns['classes'])} - {patterns['classes'][:5]}")
    print(f"   ✓ Functions: {len(patterns['functions'])} - {patterns['functions'][:5]}")
    print(f"   ✓ Modules: {len(patterns['modules'])} - {patterns['modules'][:5]}\n")

    # STEP 6: Build prompt with ranked context
    prompt = f"""You are a snnTorch code generation assistant.

USER REQUEST: {question}

RANKED DOCUMENTATION (most relevant first):

{snn_context[:8000]}

EXTRACTED PATTERNS:
- Classes: {', '.join(patterns['classes'][:10])}
- Functions: {', '.join(patterns['functions'][:10])}
- Modules: {', '.join(patterns['modules'][:10])}

Generate complete, working snnTorch code that addresses the request.

Return JSON: {{"code": "...", "confidence": 0.95, "queries_used": 10}}
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
    print("Raw LLM response:\n", response.content[:1000], "...\n")
    print("✓ Parsing response...")

    try:
        content = safe_llm_json(response.content)
        code = content.get("code", "")
        confidence = content.get("confidence", 0.0)
        print(f"   ✓ Confidence: {confidence*100:.0f}%")
        print(f"   ✓ Code length: {len(code)} characters\n")
        return {"code": code, "confidence": confidence, "queries_used": 10}
    except Exception as e:
        print(f"   ❌ Semantic parse error: {e}\n")
        return {"code": f"ERROR: {str(e)}", "confidence": 0.0, "queries_used": 10}









# ------------------ Specialized Agent: nni_agent ------------------
# ============================================================================
# IMPROVED NNI_AGENT FUNCTION ONLY - DROP-IN REPLACEMENT
# ============================================================================


def nni_agent(question: str, config: RunnableConfig, state: SummaryState):
    """
    Generates NNI experiment configuration with FULLY INTEGRATED response parsing.
    Everything is inline - no separate helper functions.
    Uses the EXACT SAME PATTERN as snnTorch agent.
    """
    
    print("=" * 80)
    print("NNI AGENT - WITH INTEGRATED PARSING")
    print("=" * 80)
    
    # ============================================================
    # STEP 1: Initialize vectorstore
    # ============================================================
    
    try:
        embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
    except:
        embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    
    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name="nni-docs",
            persist_directory="./chroma_nni_docs"
        )
        print("[NNI] ✓ Vectorstore initialized")
    except Exception as e:
        print(f"[NNI] ⚠️ Warning: Could not load vectorstore: {e}")
        vectorstore = None
    
    # ============================================================
    # STEP 2: Generate optimized search queries (INLINE)
    # ============================================================
    
    print("[NNI] Generating optimized search queries...")
    
    query_generation_prompt = f"""You are an expert at generating comprehensive search queries for NNI documentation retrieval.

Your task: Given a user's question, generate 10 DIVERSE, SPECIFIC search queries that will retrieve all relevant documentation needed to answer the question.

Requirements:
1. Each query should target a DIFFERENT aspect of the problem
2. Queries should be specific (include NNI concepts, parameter names, concrete patterns)
3. Order queries by importance (most critical first)
4. Avoid duplicate queries
5. Cover configuration, search spaces, tuning strategies, and training integration

User Question: {question}

Format: Return ONLY a JSON array of query strings:
["query1", "query2", "query3", ...]
"""
    
    generate_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )
    
    search_queries = []
    try:
        gen_response = generate_agent.run(query_generation_prompt)
        # ✓ KEY: Always extract .content first (LIKE snnTorch)
        gen_content = gen_response.content.strip()
        if gen_content.startswith('```'):
            gen_content = re.sub(r'```[a-zA-Z]*', '', gen_content)
            gen_content = re.sub(r'```', '', gen_content)
            gen_content = gen_content.strip()
        search_queries = json.loads(gen_content)
        
        if not isinstance(search_queries, list) or len(search_queries) == 0:
            raise ValueError("Empty or invalid query list")
        
        print(f"[NNI] ✓ Generated {len(search_queries)} search queries")
        for i, q in enumerate(search_queries, 1):
            print(f"  [{i}] {q[:60]}...")
    
    except Exception as e:
        print(f"[NNI] ❌ Query generation failed: {e}")
        search_queries = [
            "NNI experiment config YAML example using TPE tuner total_trials 20 parallel_trials 2 time_limit 2h",
            "NNI TPE tuner search space definition learning_rate categorical [0.0001,0.0005,0.001,0.01]",
            "NNI TPE tuner search space batch_size integer [32,64,128,256]",
            "NNI TPE tuner search space hidden_units integer [64,128,256,512]",
            "NNI uniform search space dropout_rate 0.1-0.5 example definition",
            "NNI uniform search space weight_decay 0.0-0.01 example definition",
            "Using nni.get_next_parameter() in train.py to load hyperparameters",
            "Reporting epoch accuracy to NNI via nni.report_intermediate_result in train.py",
            "Integrating model.py with NNI hyperparameters and training loop",
            "NNI tuning workflow architecture: tuner, trainer, experiment, search space overview"
        ]
    
    # ============================================================
    # STEP 3: Rank queries by relevance (INLINE)
    # ============================================================
    
    print("[NNI] Ranking queries by relevance...")
    
    ranking_prompt = f"""You are an expert at ranking search query relevance.

User Question: {question}

Candidate Queries:
"""
    for i, query in enumerate(search_queries, 1):
        ranking_prompt += f"{i}. {query}\n"
    
    ranking_prompt += """Your task: Rank these queries by RELEVANCE to answering the user's question.

Return ONLY a JSON object:
{
  "ranked_queries": ["most_relevant_query", "second_most_relevant", ...],
  "reasoning": "Brief explanation of ranking"
}

Focus on:
- Core NNI concepts mentioned in the question
- Implementation details needed
- Configuration and training requirements
"""
    
    rank_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )
    
    ranked_queries = search_queries
    try:
        rank_response = rank_agent.run(ranking_prompt)
        # ✓ KEY: Always extract .content first (LIKE snnTorch)
        rank_content = rank_response.content.strip()
        if rank_content.startswith('```'):
            rank_content = re.sub(r'```[a-zA-Z]*', '', rank_content)
            rank_content = re.sub(r'```', '', rank_content)
            rank_content = rank_content.strip()
        rank_result = json.loads(rank_content)
        ranked_queries = rank_result.get("ranked_queries", search_queries)
        reasoning = rank_result.get("reasoning", "")
        
        print(f"[NNI] ✓ Ranked {len(ranked_queries)} queries")
        print(f"[NNI] Reasoning: {reasoning}")
        for i, q in enumerate(ranked_queries, 1):
            print(f"  [{i}] {q[:60]}...")
    
    except Exception as e:
        print(f"[NNI] ❌ Ranking failed: {e}")
        ranked_queries = search_queries
    
    # ============================================================
    # STEP 4: Multi-pass retrieval from vectorstore (INLINE)
    # ============================================================
    
    nni_context = ""
    total_docs = 0
    
    if vectorstore:
        print("[NNI] Performing multi-pass retrieval...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        for i, query in enumerate(ranked_queries, 1):
            try:
                docs = retriever.get_relevant_documents(query)
                total_docs += len(docs)
                print(f"[NNI]    {i}. Query: {query[:40]}... → {len(docs)} docs")
                
                for j, doc in enumerate(docs[:3], 1):
                    source = doc.metadata.get("source", "unknown")
                    nni_context += f"\n[Ref {i}.{j} - {source}]\n{doc.page_content[:800]}\n---"
            
            except Exception as e:
                print(f"[NNI] ⚠️ Error retrieving documents for query {i}: {e}")
        
        print(f"[NNI] ✓ Total context: {len(nni_context)} chars from {len(ranked_queries)} queries")
    
    # ============================================================
    # STEP 5: Extract patterns from context (INLINE)
    # ============================================================
    
    print("[NNI] Extracting patterns from context...")
    
    patterns_choice = list(set(re.findall(r"'_type':\s*'choice'[^}]*\[([^\]]+)\]", nni_context)))[:5]
    patterns_quniform = list(set(re.findall(r"'_type':\s*'quniform'[^}]*\[([^\]]+)\]", nni_context)))[:5]

    
    print(f"[NNI]    ✓ Found {len(patterns_choice)} choice patterns")
    print(f"[NNI]    ✓ Found {len(patterns_quniform)} quniform patterns")
    
    # ============================================================
    # STEP 6: Build generation prompt (INLINE)
    # ============================================================
    
    print("[NNI] Building generation prompt...")
    
    prompt = f"""You are an NNI configuration expert.

TASK: Generate production-grade NNI Python configuration.

USER REQUEST: {question}

REFERENCE DOCUMENTATION (most relevant first):
{nni_context[:3000]}

GENERATION REQUIREMENTS:
1. Generate COMPLETE Python scripts with TWO sections:

   SECTION 1 (config.py):
   - Define search_space dict with 8 hyperparameters (mix choice and quniform)
   - Export search_space to JSON file before experiment
   - Set up argparse with 6 arguments
   - Create ExperimentConfig with tuner, assessor, training_service
   - Use AlgorithmConfig for TPE tuner and assessor
   - Use LocalConfig for GPU execution

   SECTION 2 (train.py):
   - Use nni.get_next_parameter() (v2.0 API)
   - Retrieve all 8 hyperparameters
   - Implement complete training loop with validation
   - Report intermediate results with nni.report_intermediate_result()
   - Report final result with nni.report_final_result()
   - Include argparse integration
   - Load MNIST data
   - Define PyTorch model
   - Full training with optimizer and scheduler

2. FORMAT OUTPUT:
   - Start each file with: FILE filename.py
   - Include complete, working Python code
   - NO YAML, NO hardcoded values, NO separate files
   - Strictly follow NNI v2.0 API

3. RETURN ONLY JSON:
   {{
     "code": "FILE config.py\\n...complete code...\\nFILE train.py\\n...complete code...",
     "summary": "Brief description of generated configuration"
   }}

NO MARKDOWN, NO EXPLANATIONS - ONLY VALID JSON WITH CODE KEY.
"""
    
    print("[NNI] ✓ Prompt ready")
    
    # ============================================================
    # STEP 7: Call LLM for code generation (INLINE WITH FIX)
    # ============================================================
    
    print("[NNI] Calling LLM for code generation...")
    
    code_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )
    
    response_text = ""
    try:
        response = code_agent.run(prompt)
        # ✓ CRITICAL FIX: Always access .content first (LIKE snnTorch)
        response_text = response.content
        print(f"[NNI] ✓ LLM response received ({len(response_text)} chars)")
    
    except Exception as e:
        print(f"[NNI] ❌ LLM call failed: {e}")
        return {"code": "", "error": str(e)}
    
    # ============================================================
    # STEP 8: Parse response with integrated extraction (INLINE)
    # ============================================================
    
    print("[NNI] Parsing response...")
    
    code = ""
    try:
        # ✓ CRITICAL FIX: Use safe JSON extraction (LIKE snnTorch)
        parse_content = response_text.strip()
        if parse_content.startswith('```'):
            parse_content = re.sub(r'```[a-zA-Z]*', '', parse_content)
            parse_content = re.sub(r'```', '', parse_content)
            parse_content = parse_content.strip()
        
        response_data = json.loads(parse_content)
        code = response_data.get("code", "")
        
        if not code:
            print("[NNI] ⚠️ No code in response")
            return {"code": "", "error": "Empty code in response"}
        
        print(f"[NNI] ✓ Extracted code: {len(code)} chars")
    
    except json.JSONDecodeError as e:
        print(f"[NNI] ❌ JSON parse error: {e}")
        print(f"[NNI] Response text: {response_text[:500]}")
        return {"code": "", "error": str(e)}
    
    # ============================================================
    # STEP 9: Unescape newlines and handle escaping (INLINE)
    # ============================================================
    
    print("[NNI] Unescaping newlines...")
    
    code = code.replace("\\n", "\n")
    code = code.replace("\\t", "\t")
    code = code.replace('\\"', '"')
    print(f"[NNI] ✓ Code unescaped: {len(code)} chars")
    
    # ============================================================
    # STEP 10: Validate extracted code (INLINE)
    # ============================================================
    
    print("[NNI] Validating extracted code...")
    
    validation_checks = {
        "has_file_markers": "FILE" in code,
        "has_search_space": "search_space" in code and "_type" in code,
        "has_nni_integration": "nni.get_next_parameter" in code,
        "has_experiment_config": "ExperimentConfig" in code,
        "has_training_loop": "for epoch" in code or "for batch" in code,
        "has_torch_import": "import torch" in code,
    }
    
    passed = sum(1 for v in validation_checks.values() if v)
    total = len(validation_checks)
    score = int(100 * passed / total)
    
    print(f"[NNI] Validation score: {score}% ({passed}/{total} checks passed)")
    
    for check_name, result in validation_checks.items():
        status = "✓" if result else "✗"
        print(f"[NNI] {status} {check_name}")
    
    # ============================================================
    # STEP 11: Extract FILE sections (INLINE)
    # ============================================================
    
    print("[NNI] Extracting FILE sections...")
    
    pattern = r"FILE\s+(\S+\.py)\s*\n(.*?)(?=FILE|\Z)"
    matches = list(re.finditer(pattern, code, re.DOTALL | re.MULTILINE))
    
    files = {}
    for i, match in enumerate(matches, 1):
        filename = match.group(1)
        file_content = match.group(2).strip()
        files[filename] = file_content
        print(f"[NNI] {i}. {filename} - {len(file_content)} chars")
    
    if not files:
        print("[NNI] ⚠️ No FILE sections found, returning code as-is")
        files["generated_code.py"] = code
    
    print("=" * 80)
    print("NNI AGENT COMPLETE")
    print("=" * 80)
    
    return {
        "code": code,
        "validation_score": score,
        "checks": validation_checks,
        "files": files,
    }







# ------------------ General Code Generation Function ------------------
def generate_code(config: RunnableConfig, state: SummaryState):
    """
    Generate code using a two-phase approach:
    1. Orchestrator decides which tools to call
    2. Orchestrator integrates the tool outputs into a single coherent solution
    """
    print("---GENERATING CODE SOLUTION---")
    print("--- PHASE 1: TOOL SELECTION AND EXECUTION ---")
    
    question = state.research_topic
    if isinstance(question, list):
        question_str = " ".join(question)
    else:
        question_str = str(question)
    
    # Define available tools
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
        - For each tool, provide a specific query
        - Each tool MUST return code formatted with # FILE: markers
        - Respond with ONLY a JSON object containing tool calls

        Available tools:
        1. snnTorch_agent: Generates SNN model code
        - Output format: Dict with "code" key containing Python code
        - MUST include # FILE: markers for each module/file
        - Example output structure:
            # FILE: model.py
            [SNN model code]
            
            # FILE: utils.py
            [helper functions]

        2. nni_agent: Generates NNI experiment configuration
        - Output format: Dict with "code" key containing Python code
        - MUST include # FILE: markers for each component
        - Example output structure:
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
            "expected_files": ["model.py", "utils.py"],
            "format_requirement": "MUST include # FILE: markers for each file"
            },
            {
            "name": "nni_agent",
            "query": "specific question for NNI setup",
            "expected_files": ["config.py", "train.py"],
            "format_requirement": "MUST include # FILE: markers for each file"
            }
        ]
        }

        Critical Rules:
        - Only include tools that are needed for the user request
        - Each query should be focused and specific
        - Do NOT generate code yourself - only decide which tools to call
        - Output ONLY valid JSON (no markdown code blocks, no explanations)
        - You are responsible for ensuring each tool gets a well-formed query
        """

    
    prompt = (
        f"Context (previous code):\\n{state.code}\\n\\n"
        f"User request:\\n{question_str}\\n\\n"
        f"Instructions: {tool_selection_prompt}"
    )
    
    # Get tool selection from orchestrator
    response = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=True,
        use_json_mode=True,
    ).run(prompt)
    
    print("ORCHESTRATOR SPECIALIZED AGENTS SELECTION:\\n", response.content)
    
    # Parse and execute tool calls
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
        
        # Execute the tool
        result = None
        if tool_name == "snnTorch_agent":
            result = snnTorch_agent(query, config, state)
        elif tool_name == "nni_agent":
            result = nni_agent(query, config, state)
        else:
            print(f"   ❌ Unknown tool: {tool_name}")
            continue
        
        if not result:
            print(f"   ❌ Tool returned empty result")
            continue
        
        # Extract code from result
        code = result.get("code", "") if isinstance(result, dict) else str(result)
        
        # VALIDATION: Check FILE marker format
        import re
        file_markers = re.findall(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', code, re.MULTILINE)
        
        print(f"   ✓ Received {len(code)} characters of code")
        
        if len(file_markers) == 0:
            print(f"   ⚠️  WARNING: No # FILE: markers found!")
            print(f"   WRAPPING output with default marker...")
            code = f"# FILE: {tool_name.lower()}_output.py\n{code}"
            file_markers = [f"{tool_name.lower()}_output.py"]
        
        print(f"   ✓ Detected {len(file_markers)} file(s): {file_markers}")
        tool_outputs[tool_name] = code

    if not tool_outputs:
        print("\n❌ ERROR: No code was generated from any tool")
        return "Error: No code generated from tools"

    print(f"\n✅ Collected {len(tool_outputs)} tool outputs ready for integration")

    
    print(f"\\n--- PHASE 2: CODE INTEGRATION ---")
    print(f"Integrating {len(tool_outputs)} code components...")
    
    # INTEGRATION STEP: Have orchestrator combine the code
    integration_prompt = f"""You are a code integration expert. You have received code from multiple specialized agents.

    Your task:
    - Analyze all the code components provided
    - Create a SINGLE, COHERENT, INTEGRATED code solution
    - Ensure proper imports, no duplicates, correct dependencies
    - Make sure all components work together seamlessly
    - Add necessary glue code to connect the components

    IMPORTANT: Structure your code with file markers like this:

    # FILE: model.py
    [model code here]

    # FILE: config.py
    [config code here]

    # FILE: training.py
    [training code here]

    Start each new file section with: # FILE: filename.py

    User's original request:
    {question_str}

    Code components received:
    """
    
    # Add each tool output to the prompt
    for tool_name, code in tool_outputs.items():
        integration_prompt += f"\\n\\n=== Code from {tool_name} ===\\n{code}\\n"
    
    integration_prompt = """You are a code integration expert with STRICT FORMATTING requirements.

        Your PRIMARY task:
        - Integrate multiple code components into ONE coherent solution
        - Structure output with # FILE: markers for each distinct file
        - Ensure all files work together seamlessly

        CRITICAL FORMATTING RULES (MUST FOLLOW):

        1. OUTPUT STRUCTURE:
        Every file MUST start with: # FILE: filename.py
        
        Example:
        # FILE: model.py
        import torch
        import snntorch as snn
        
        class SNNModel(torch.nn.Module):
            pass
        
        # FILE: config.py
        LEARNING_RATE = 0.001
        
        # FILE: training.py
        from model import SNNModel
        from config import LEARNING_RATE

        2. FILE MARKERS:
        - Format: "# FILE: filename.py" (with exactly one space after #)
        - Must be at start of line (no indentation)
        - filename must include .py extension
        - Each new file starts with its own marker
        - NEVER reuse markers for the same file

        3. FILE ORGANIZATION:
        - Arrange files in logical order (dependencies first):
            1. config.py or constants
            2. model.py or classes
            3. utils.py or helper functions
            4. main.py or training.py (entry point last)
        
        4. IMPORTS:
        - Consolidate all imports at the top of each file
        - Use relative imports between your generated files
        - Example: from config import PARAM
        - Do NOT repeat imports across files

        5. OUTPUT CONSTRAINTS:
        - Output ONLY Python code with # FILE: markers
        - NO markdown code fences (```python, ``` etc)
        - NO section headers (===, ---, etc)
        - NO explanations or comments outside code
        - NO "Here is the integrated code" preamble
        - Start directly with "# FILE: first_file.py"

        6. VALIDATION:
        - Each file must be self-contained (can understand imports)
        - NNI configurations must reference correct model class names
        - All imports must be available (standard library or mentioned in requirements)
        - Clear dependencies documented in comments

        USER REQUEST: {question_str}

        CODE COMPONENTS TO INTEGRATE:
        """
            
        # Add each tool output
    for tool_name, code in tool_outputs.items():
        integration_prompt += f"\n=== CODE FROM {tool_name} ===\n{code}\n"

        integration_prompt += """

        FINAL INTEGRATION CHECKLIST:
        □ All files have "# FILE: filename.py" markers
        □ Files ordered by dependency
        □ No duplicate imports
        □ Relative imports between files work
        □ Model classes match NNI config references
        □ All files present as expected

        Remember: Your output is parsed programmatically.
        Any deviation from # FILE: format will break the code execution system.

        Output ONLY the integrated code. Start with "# FILE: " immediately.
        """

    
    # Get integration response
    integration_response = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
    ).run(integration_prompt)

    integrated_code = integration_response.content

    # ============================================================
    # VALIDATION: Check integration output format
    # ============================================================

    import re

    print("\n" + "="*80)
    print("INTEGRATION OUTPUT VALIDATION")
    print("="*80)

    # Extract all file markers
    file_markers = re.findall(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', integrated_code, re.MULTILINE)

    print(f"\n📊 Files detected: {len(file_markers)}")
    for i, fname in enumerate(file_markers, 1):
        print(f"   {i}. {fname}")

    # Validation checks
    checks = {
        "has_file_markers": len(file_markers) > 0,
        "files_have_py_extension": all(f.endswith('.py') for f in file_markers),
        "unique_filenames": len(file_markers) == len(set(file_markers)),
        "has_model_file": any('model' in f.lower() for f in file_markers),
        "starts_with_file_marker": integrated_code.lstrip().startswith("# FILE:"),
    }

    passed = sum(1 for v in checks.values() if v)
    score = int((passed / len(checks)) * 100)

    print(f"\n✓ Validation score: {score}% ({passed}/{len(checks)} checks passed)")

    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")

    # Warnings for common issues
    if len(file_markers) == 0:
        print("\n⚠️  CRITICAL: No # FILE: markers found!")
        print("   Attempting to wrap output...")
        integrated_code = f"# FILE: main.py\n{integrated_code}"

    if "```" in integrated_code:
        print("\n⚠️  WARNING: Found markdown code fences (```)")
        print("   Removing markdown formatting...")
        integrated_code = re.sub(r"```[^`]*\n?", "", integrated_code)

    if not integrated_code.lstrip().startswith("# FILE:"):
        print("\n⚠️  WARNING: Output doesn't start with # FILE: marker")
        print("   This may cause parsing issues!")

    print(f"\nFinal code length: {len(integrated_code):,} characters")
    print("="*80 + "\n")

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

    output_directory = "./generated_code"
    #output_filename = "latest_generated_code.py"  

    # Construct unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_code_{timestamp}.py"

    # To overwrite the file each time:
    save_code_to_file(code, output_directory, output_filename, mode="w")

    state.code = code
    #state.fixed_code = code # save for later sandboxing

    print("CODE SOLUTION:\n", code)
    

    

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
    timeout_ms: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Upload multi-file code to E2B sandbox and execute.
    
    Args:
        code_str: Generated code with # FILE: markers
        main_file: Entry point filename (auto-detected if None)
        install_packages: List of packages to pip install
        timeout_ms: Execution timeout in milliseconds
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
        sandbox.commands.run(f"mkdir -p {upload_dir}", timeout_ms=0)
        
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
        #     timeout_ms=timeout_ms
        # )

        exec_result = sandbox.commands.run(
        f"python -c \"import sys; sys.path.insert(0, '/home/user'); exec(open('/home/user/{mainfile}').read())\"",
        timeout_ms=timeout_ms
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
        timeout_ms=120000,
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
        "sandbox_feedback_pyright": static_result or "No result",
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
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            response_dict = json.loads(json_match.group())
        else:
            response_dict = json.loads(response.content)
        
        decision = response_dict.get('response', 'approve')
        file_name = response_dict.get('file_name', None)  # ← Gets filename if present
        
        print(f"Decision: {decision}")
        if file_name:
            print(f"File to execute: {file_name}")
        
        # Store in state
        state.user_feedback_processed = decision
        state.user_specified_file = file_name  # ← NEW: Store the filename
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Raw response: {response.content}")
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

