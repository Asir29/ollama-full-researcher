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