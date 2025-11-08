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

from helper import (
    load_and_split,
    build_vectorstore,
    extract_packages_from_imports,
    remove_think_tags,
    save_code_to_file
)



# -------------------------
# CopilotKit / LangGraph Utilities
# -------------------------
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_exit, copilotkit_customize_config











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