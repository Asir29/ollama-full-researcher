# langgraph dev --no-reload
# to run without reloading

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
from scholarly import scholarly                        # Google Scholar search
from semanticscholar import SemanticScholar            # Semantic Scholar API

# -------------------------
# HTML / Web Parsing
# -------------------------
from bs4 import BeautifulSoup  # HTML parsing and cleaning

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
    code_reflection_instructions, code_search_instructions
)

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

    print(f"Current state: {state}")

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
    print(f"Current state: {state}")
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
    
    configurable = Configuration.from_runnable_config(config)
    # Search the web
    agent = Agent(
        model=Ollama(id="mistral:latest"),
        tools=[GoogleSearchTools()],
        show_tool_calls=False,
        markdown=True
        
    )

    query = web_search_instructions + "\n Search for the following query: " + state.search_query + "\n"
    #run_response = agent.run(query)
    #run_response = await agent.arun(query)  # ✅ Non-blocking!
    run_response = await asyncio.to_thread(lambda: agent.run(query))

    content = run_response.content

    return {"raw_search_result": content}


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

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Sources",
        "content": running_summary
    }))
    return {"running_summary": running_summary}






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

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

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

    

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

async def finalize_summary(state: SummaryState, config: RunnableConfig):
    """ Finalize the summary """
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Finalize Summary",
        "content": "Finalizing research summary..."
    }))
    
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    final_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    
    await copilotkit_emit_message(config, json.dumps({
        "node": "Finalize Summary",
        "content": final_summary
    }))
    
    # Signal completion to copilotkit
    await copilotkit_exit(config)
    
    return {"running_summary": final_summary}

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
    print(f"Current state: {state}")
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






async def academic_research(state, config, papers_limit=3):
    query = state.get("search_query") if isinstance(state, dict) else getattr(state, "search_query", "")

    # scholarly is synchronous, so run in a thread to avoid blocking event loop
    def fetch_papers():
        search_query = scholarly.search_pubs(query)
        results = []
        for _ in range(papers_limit):
            try:
                paper = next(search_query)
                title = paper.get('bib', {}).get('title', 'No title')
                abstract = paper.get('bib', {}).get('abstract', 'No abstract')
                results.append(f"{title} - {abstract}")
            except StopIteration:
                break
        return results

    abstracts = await asyncio.to_thread(fetch_papers)

    return {"academic_source_content": abstracts}



    

    



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

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    await copilotkit_emit_message(config, json.dumps({
        "node": "Summarize Sources",
        "content": running_summary
    }))
    return {"running_summary": running_summary}


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
    Use LLM to extract pip-installable packages from code block as a comma-separated string.
    Ignores imports inside Python code fences.
    """
    print("Extracting packages from imports...", import_block)

    # Remove any fenced code block
    pattern = r"```.*?```"
    cleaned_import_block = re.sub(pattern, "", import_block, flags=re.DOTALL).strip()


    prompt_template = """
        You are a Python code analyzer.

        Your task: Extract ONLY the top-level pip-installable package names from the given Python code. 
        Return them as a single comma-separated string, and nothing else. Do NOT include explanations, markdown, quotes, code fences, or anything extra.

        ### Strict Rules:
        1. Include ONLY top-level packages that are imported in the code.
        - Example: `import numpy as np` → numpy
        - Example: `from torch.nn import Linear` → torch
        2. Exclude:
        - Python standard library modules (sys, os, typing, json, re, etc.)
        - Relative imports (e.g., `from .utils import helper`)
        - Local files or scripts
        3. Preserve the order in which packages appear.
        4. Ignore any import statements inside code fences (```).
        5. Output ONLY the comma-separated package names.

        ### Few-shot Examples:

        #### Example 1:
        Python Code:

        import torch
        import os
        import numpy as np
        from torch.nn import Linear

        Output:
        torch,numpy

        #### Example 2:
        Python Code:

        from sklearn.model_selection import train_test_split
        import pandas as pd
        import sys

        Output:
        sklearn,pandas

        #### Example 3:
        Python Code:

        import requests
        import json
        from .my_module import helper

        Output:
        requests

        ### Python Code to Analyze:
        {import_block}
        """





    prompt = prompt_template.format(import_block=cleaned_import_block)

    agent = Agent(
        model=Ollama(id="mistral:latest"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=False,
    )

    try:
        response = agent.run(prompt)
        print("RAW RESPONSE TEXT PACKAGES EXTRACTOR:\n", response)

        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse into clean list
        package_list = [pkg.strip() for pkg in response_text.split(",") if pkg.strip()]
        return package_list

    except Exception as e:
        print("Fallback to empty package list due to error:", e)
        return []



def search_relevant_sources(state: SummaryState, config: RunnableConfig):
    """
    Crawls the SNN documentation site and creates a persisted vectorstore.
    """

    print("---CREATING SNN VECTORSTORE---")

    # 1️⃣ Crawl SNN docs
    loader = RecursiveUrlLoader(
        url="https://snntorch.readthedocs.io/en/latest/",
        max_depth=3,
        extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
    )
    snn_docs = loader.load()
    print(f"Loaded {len(snn_docs)} SNN docs pages")

    # 2️⃣ Split documents
    doc_splits = load_and_split(snn_docs, chunk_size=512, overlap=50)

    # 3️⃣ Create & persist vectorstore
    embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
    vectorstore = build_vectorstore(
        doc_splits,
        embedding_model,
        collection="snntorch-docs",
        persist="./chroma_snn_docs"
    )
    vectorstore.persist()
    print("SNN vectorstore persisted at './chroma_snn_docs'")

    return {"status": "vectorstore_created"}




def generate_code(question: str, config: RunnableConfig, state: SummaryState):


    print("--- GENERATING CODE ---")

    # -------------------------
    # 1️⃣ Web Search for User URLs
    # -------------------------
    # Use search tools
    duck_tool = DuckDuckGoTools(fixed_max_results=3)
    google_tool = GoogleSearchTools(proxy=None)  # or proxy="http://your_proxy:port"
    baidusearch_tool = BaiduSearchTools(fixed_max_results=3)

    search_agent = Agent(
        model=Ollama(id="qwen3:latest"),
        tools=[google_tool],
        show_tool_calls=True,
        markdown=False
    )

    # Format search query
    search_query = code_search_instructions.format(research_topic=state.research_topic)
    search_response = search_agent.run(search_query)
    print("RAW SEARCH RESPONSE:\n", search_response)

    # Clean up <think> tags from Deepseek/LLM responses
    content = search_response.content
    while "<think>" in content and "</think>" in content:
        start = content.find("<think>")
        end = content.find("</think>") + len("</think>")
        content = content[:start] + content[end:]

    # Parse URLs returned by the search
    try:
        data = json.loads(content or "{}")
        urls = data.get("urls", [])
    except json.JSONDecodeError:
        urls = []
    print(f"User URLs: {urls}")

    # Load content from user-specified URLs
    url_docs = []
    for url in urls:
        url_docs.extend(WebBaseLoader(url).load())
    url_context = "\n\n".join([doc.page_content for doc in url_docs])

    # -------------------------
    # 2️⃣ Load SNN Documentation Vectorstore
    # -------------------------
    embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")
    #embedding_model = SentenceTransformerEmbeddings("nomic-ai/CodeRankEmbed")

    # Load persisted vectorstore
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="snntorch-docs",
        persist_directory="./chroma_snn_docs"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve relevant SNN docs
    snn_relevant = retriever.get_relevant_documents(state.research_topic)
    snn_context = "\n\n".join([doc.page_content for doc in snn_relevant])

    # -------------------------
    # 3️⃣ Combine contexts
    # -------------------------
    combined_context = snn_context + "\n\n" + url_context

    # -------------------------
    # 4️⃣ Construct prompt for code generation
    # -------------------------
    code_agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    prompt = (
    code_assistant_instructions
    + "\nBased on the following context, generate the code that satisfies the question:\n"
    + "Context:\n" + combined_context
    + "\nQuestion:\n" + state.research_topic
    + "\nIf the question asks for a modification to previously generated code, "
      "return the ENTIRE codebase again, with the requested modification fully integrated. "
      "Do NOT output only the new fragment—always output the complete updated code.\n"
    + "Here is the previously generated code:\n"
    + state.code
    )


    # Run the code generation agent
    code_response = code_agent.run(prompt)
    print("RAW CODE RESPONSE:\n", repr(code_response.content))
    content = code_response.content
    code = ""

    try:
        content = json.loads(content or "{}")
        code = content.get("code", "")
    except json.JSONDecodeError:
        code = "Error: Unable to parse code response."
        
    print("PARSED CODE:\n", code)
    #state.code = code
    #state.fixed_code = code # save for later sandboxing

    return code
    




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
    print(f"Current state: {state}")

    # State
    messages = state.messages
    iterations = state.code_iterations

    # Solution
    code = generate_code(messages, config, state)

    state.code = code
    #state.fixed_code = code # save for later sandboxing

    print("CODE SOLUTION:\n", code)
    

    

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











def check_code_sandbox(state: SummaryState, config: RunnableConfig):
    """
    Check code in a sandbox with dependency installation.
    """
    print("---CHECKING CODE IN SANDBOX---")
    #agent that extracts ONLY the imports

    code = ""

    print("state in check_code_sandbox: ", state)

    print("Current state.user_feedback_processed: ", state.user_feedback_processed)
    if state.user_feedback_processed == "evaluation":
        code = state.normalized_code # from normalization step
    elif state.user_feedback_processed == "execute":
        code = state.code # from code generation step

    agent = Agent(
        model=Ollama(id="mistral:latest"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=False,
    )
    
    imports_extractor_instructions = """
    You are given a block of Python code.
    Your task is to return ONLY the import statements as a single block of text.
    ### Instructions:
    - Output ONLY the import statements, preserving their order, Separated by new lines.
    - Do NOT include explanations, markdown, quotes, or code fences.
    - Exclude any comments or non-import lines.

    - example:
        import torch
        import snntorch as snn
        
    """
    print("Code to extract imports from:\n", code)
    query = imports_extractor_instructions + "\n Code:\n" + code + "\n"

    response = agent.run(query)

    
    imports = response.content
    
    print("Imports to give to extract_packages:", imports)
    #code = state.code_generation.code or ""
    
    cleaned_code = code.replace("```python\n", "").replace("\n```", "") # remove markdown formatting if any
    #combined_code = f"{imports}\n{code}"
    


    #sandbox_pyright = Sandbox.create("OpenEvalsPython") # already with pyright and uv installed
    static_evaluator = create_pyright_evaluator()


    sandbox_execution = Sandbox.create() # with this use sbx.run_code

    # Static type check
    
    static_evaluation_result = static_evaluator(outputs=cleaned_code)
    
 

    
    # Extract dependencies via LLM
    try:
        packages = extract_packages_from_imports(imports)
        print("Inferred packages:", packages)
    except Exception as e:
        print("Fallback to empty package list due to error:", e)
        packages = []

    # Install packages
    for pkg in packages:
        print(f"Installing {pkg} in sandbox...")

        if pkg in ["pytorch","torch"]:
            sandbox_execution.commands.run("pip install torch --index-url https://download.pytorch.org/whl/cpu", timeout=0) # cpu version of  torch, lighter (the sandbox is 1GB)
        elif pkg in ["tensorflow"]:
            sandbox_execution.commands.run("pip install --no-cache-dir tensorflow-cpu", timeout=0) # cpu version of tensorflow, lighter
        elif pkg in ["sklearn"]:
            sandbox_execution.commands.run("pip install scikit-learn", timeout=0) # sklearn is actually scikit-learn
        elif pkg in ["opencv", "cv2"]:
            sandbox_execution.commands.run("pip install opencv-python-headless", timeout=0)
        elif pkg in ["PIL"]:
            sandbox_execution.commands.run("pip install Pillow", timeout=0)
        else:
            result = sandbox_execution.commands.run(f"pip install {pkg}", timeout=0)
            print(result.stdout or result.stderr)

    print("code to execute in sandbox:\n", cleaned_code)

    # Execution check
    evaluator_exec = create_e2b_execution_evaluator(sandbox=sandbox_execution)
    eval_result_execution = evaluator_exec(outputs=cleaned_code)
    print("Execution result:", eval_result_execution)

    # Direct execution in sandbox (outside evaluator)
    run_result = sandbox_execution.run_code(cleaned_code)

    print("=== STDOUT ===")
    print(run_result)
    
    state.sandbox_execution_result = run_result
    

    return {
        "sandbox_feedback_pyright": static_evaluation_result or "No pyright result",
        "sandbox_feedback_execution": eval_result_execution,
        "sandbox_execution_result": run_result
    }





def reflection(state: SummaryState, config: RunnableConfig):
    """
    Reflect on the code and decide whether to finish or retry.
    """

    print("---REFLECTING ON CODE---")

    # Check code
    pyright_feedback = state.sandbox_feedback_pyright
    execution_feedback = state.sandbox_feedback_execution
    print("PYRIGHT FEEDBACK:", pyright_feedback)

    #if(pyright_feedback['key'] == "pyright_succeeded" and execution_feedback['key'] == "execution_succeeded"):
    #    print("---NO ERRORS---")
    #    return {"route": "no_errors"}  # Return a dictionary
    
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

    print("code before replace:", code)

    # ensure 'code' is a dict
    if isinstance(code, dict):
        code = code.get("code", "")

    code = code.replace("```python\n", "").replace("\n```", "")
    #print("Combined code for static evaluation:\n", combined_code)
    

    static_evaluation_result = static_evaluator(outputs=code)

    

    feedback_prompt = {
        "instruction": "Please review the code and choose: *approve*, *regenerate*, *evaluate with a reference code* or to *execute the code*. Then explain your decision.",
        "code": state.code,
        "static_python_check": static_evaluation_result or "No pyright result",
    }

    # This pauses execution and will show editable fields
    user_input = interrupt(feedback_prompt)

    # Return the user edits into state
    # return {
    #     "user_feedback": user_input
    # }







def process_feedback(state: SummaryState, config: RunnableConfig):
    """Process the user feedback and classify next action."""
    print("---PROCESSING FEEDBACK DECISION---")

    print("\nstate in process feedback:", state)

    agent = Agent(
        model=Ollama(id="deepseek-r1"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    prompt = f"""\

    You are a decision agent.

    You are given user feedback: {state.research_topic}

    Based on this feedback, respond with **only one** of the following exact JSON objects:

    {{"response": "approve"}}
    {{"response": "regenerate"}}
    {{"response": "evaluation"}}
    {{"response": "execute"}}

    ### Definitions:
    - Use **"evaluation"** if the user wants to perform an evaluation or if the user talks about to evaluate.
    - Use **"approve"** if the user is fully satisfied and wants to keep the code exactly as it is, with **no changes requested**.
    - Use **"regenerate"** if the user asks for **any modifications**, **improvements**, or expresses **dissatisfaction** with the current code.
    - Use **"execute"** if the user wants to run the code in a sandbox to check for errors.
    Only return the JSON object — do not include any other text, explanations, or logs.
    """


    


    try:
        response = agent.run(prompt)
        response_text = getattr(response, "content", str(response))
        print(f"RAW DECISION RESPONSE: {response_text}")

        # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
        # It appears very challenging to prompt them out of the responses 
        while "<think>" in response_text and "</think>" in response_text:
            start = response_text.find("<think>")
            end = response_text.find("</think>") + len("</think>")
            response_text = response_text[:start] + response_text[end:]

        print(f"RESPONSE TEXT: {response_text}")

        parsed = json.loads(response_text)
        action = parsed.get("response", "regenerate").lower()

        if action not in {"regenerate", "approve", "evaluation", "execute"}:
            action = "regenerate"

    except Exception as e:
        print(f"Error parsing feedback response: {e}")
        action = "regenerate"

    state.user_feedback_processed = action
    return {"user_feedback_processed": action}



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

    prompt = f"""
You are a code normalization assistant.

You are given two Python code snippets:

<<GENERATED>>
{state.code}
<<END_GENERATED>>

<<REFERENCE>>
{state.research_topic}
<<END_REFERENCE>>

Task:

- Produce ONE self-contained Python script as output.
- The script must contain two clear sections:

    ### Generated code
    - Include the generated snippet **exactly as provided**.
    - Do not modify the internal logic, classes, or functions.
    - Only wrap it so it can run on the same TOY_INPUT as the reference code.

    ### Reference code
    - Include the reference snippet.
    - If needed, adapt its input handling so it can run on the same TOY_INPUT.

- Both sections MUST use the **same TOY_INPUT**.
- At the end of the script:
    - Instantiate any classes or functions needed.
    - Run both sections on TOY_INPUT.
    - Print both outputs clearly, like:
      Generated output: <value>
      Reference output: <value>

- Do NOT include JSON, markdown fences, or explanations.
- Ensure the output is pure, executable Python code.
"""


    script_str = ""

    try:
        response = agent.run(prompt)
        script_str = getattr(response, "content", str(response))

        # Remove <think> artifacts
        # while "<think>" in script_str and "</think>" in script_str:
        #     start = script_str.find("<think>")
        #     end = script_str.find("</think>") + len("</think>")
        #     script_str = script_str[:start] + script_str[end:]

        # Remove ``` fences if the model added them
        # script_str = script_str.strip()
        # if script_str.startswith("```"):
        #     script_str = script_str.strip("`")
        #     # sometimes comes like ```python\n...\n```
        #     script_str = script_str.replace("python\n", "", 1).strip()

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

        # Remove <think> tags if present
        while "<think>" in response_text and "</think>" in response_text:
            start = response_text.find("<think>")
            end = response_text.find("</think>") + len("</think>")
            response_text = response_text[:start] + response_text[end:]

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
        Task: Modify the given PyTorch code to measure and report the following performance metrics for a single forward pass of each model:

        1. Execution time of the forward pass in seconds.
        2. Total number of trainable parameters in the model.
        3. Memory usage (approximate GPU memory if on CUDA, otherwise CPU memory) during the forward pass.

        The code should work for both standard nn.Module and custom spiking neuron models. Add metrics printing after running the forward pass. Keep the existing functionality intact.

        Requirements:
        - Use torch.cuda.memory_allocated() and torch.cuda.reset_peak_memory_stats() for GPU memory tracking if CUDA is available; otherwise, use CPU memory via tracemalloc or similar.
        - Measure time using time.time() before and after the forward pass.
        - Count parameters using sum(p.numel() for p in model.parameters()).
        - Print a summary like:
        Model: MyNet
        Forward pass time: 0.0123 s
        Total parameters: 1_234
        Peak memory usage: 12.3 MB
        - Ensure the metrics are accurate for one forward pass, not training steps or multiple passes.

        Modify only what is needed to collect these metrics; do not change model architectures or forward computations.
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


# Add nodes for code generation and code checking
builder.add_node("search_relevant_sources", search_relevant_sources)

builder.add_node("generate", generate)  # generation solution
#builder.add_node("check_code", code_check)  # check code
#builder.add_node("continue_generation", continue_generation)
builder.add_node("collect_feedback", collect_feedback)
builder.add_node("process_feedback", process_feedback)
builder.add_node("check_code_sandbox", check_code_sandbox)  # check code in sandbox
builder.add_node("reflection", reflection)  # reflect on code errors
#builder.add_node("evaluation", evaluation) # perform a systematic evaluation with groud truth code

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
builder.add_edge("academic_research", "summarize_academic_sources")
builder.add_edge("summarize_academic_sources", END)

builder.add_edge("search_relevant_sources", "generate")

#builder.add_edge("generate", "check_code_sandbox")
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
#builder.add_edge("evaluation", END)
builder.add_edge("code_normalization", "collect_feedback_normalization")
builder.add_edge("collect_feedback_normalization", "process_feedback_normalization")
#builder.add_edge("process_feedback_normalization", "check_code_sandbox")
builder.add_edge("process_feedback_normalization", "add_performance_metrics")
builder.add_edge("add_performance_metrics", "check_code_sandbox")



# Add memory saver for checkpointing
memory = MemorySaver()

# Compile graph with checkpointing and interrupts
graph = builder.compile(
#checkpointer=memory,
#interrupt_after=[""],
)

