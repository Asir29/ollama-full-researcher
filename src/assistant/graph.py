# langgraph dev --no-reload
# to run without reloading

import torch
torch.cuda.empty_cache()


import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(method)s %(path)s %(status)s",  # Customize the format
)



logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from src.assistant.configuration import Configuration
from src.assistant.utils import deduplicate_and_format_sources, format_sources
from src.assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from src.assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, web_search_instructions, web_search_description, web_search_expected_output, router_instructions, code_assistant_instructions, academic_summarizer_instructions, code_reflection_instructions, code_search_instructions
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_exit, copilotkit_customize_config

from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
from langchain_ollama import ChatOllama

from scholarly import scholarly

from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.prompts import ChatPromptTemplate

#from utils import get_prompt_code_assistant

import re

import textwrap

from langgraph.types import Command

from pydantic import BaseModel, Field
#from langchain_core.pydantic_v1 import BaseModel, Field
from typing import TypedDict, Literal
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

import asyncio
from semanticscholar import SemanticScholar

from e2b_code_interpreter import Sandbox
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator
from openevals.code.e2b.execution import create_e2b_execution_evaluator


from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.ollama import Ollama
import numpy as np
import json



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        HumanMessage(content=f"Input: {state.research_topic}")
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
        model=Ollama(id="mistral-nemo"),
        tools=[GoogleSearchTools()],
        show_tool_calls=False,
        markdown=True
        
    )

    query = web_search_instructions + "\n Search for the following query: " + state.search_query + "\n"
    #run_response = agent.run(query)
    run_response = await agent.arun(query)  # ✅ Non-blocking!

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




# Data model for the Code
class CodeOutput(BaseModel):
    """Code output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    

code_parser_instructions = """\
    You are a code parser. Your task is to extract the prefix, imports, and code from the following text.
    The text is a response from a code assistant. The response is structured in JSON format.
    
    The output should be a JSON object with the following fields:
    {
        "prefix": "A description of the code solution",
        "imports": "The necessary import statements",
        "code": "The functioning code block"
    }
    """





# # Proper embedding wrapper for LangChain
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name: str):
#         self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda") # change here if want to use "cuda" or "cpu"

#     def embed_documents(self, texts):
#         return self.model.encode(texts).tolist()

#     def embed_query(self, text):
#         return self.model.encode([text])[0].tolist()
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



def generate_code(question: str, config: RunnableConfig, state: SummaryState) -> CodeOutput:
    agent = Agent(
        model=Ollama(id="codellama"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    # Load URLs
    data = json.loads(state.urls or "{}" )
    urls = data["urls"]
    print(f"URLs: {urls}")

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [doc for sublist in docs for doc in sublist]
    print(f"Loaded {len(docs_list)} documents")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=128, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Use lightweight embedding model
    embedding_model = SentenceTransformerEmbeddings("nomic-ai/CodeRankEmbed")


    # Vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        collection_name="code-rag",
        persist_directory=None  # 👈 In-memory only, no persistence. Remove the line to make the embedding persistent.
    )

    # Retrieve
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(state.research_topic)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("CONTEXT:", context)

    # Construct query
    query = code_assistant_instructions + "\n Based on the following context, generate the code that satisfies the question:" + "Context: " + context + "\nQuestion: " + state.research_topic + "\n" + "If the question is a request related to the previusly generated code, consider it when producing the response. In the following the previously generated code: " + state.imports + state.code

    # Run agent
    response = agent.run(query)
    response_text = response.content
    print("RAW RESPONSE TEXT:\n", repr(response_text))

    # Parse result
    # try:
    #     parsed_response = json.loads(response_text)
    # except json.JSONDecodeError as e:
    #     print(f"Error during JSON parsing: {e}")
    #     return CodeOutput(prefix="", imports="", code="")

    # imports_str = parsed_response.get("imports", "")
    # if isinstance(imports_str, list):
    #     imports_str = "\n".join(imports_str)

    # Parse result with LLM
    agent_parser = Agent(
        model=Ollama(id="codellama"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )  

    code_parser_instructions = """You are a code parser.

    Take the generated code response and extract the following fields:
    - prefix: a short description of the code solution
    - imports: all required import statements as a single string
    - code: the actual Python code (excluding imports)

    ⚠️ Important:
    - The imports must be returned as a single string (not a list).
    - Do not return markdown formatting like ```json or explanations.
    - Return ONLY a valid JSON object — no comments, no intro.

    Example output format:
    {
    "prefix": "A classifier using snnTorch.",
    "imports": "import torch\\nimport snntorch as snn",
    "code": "class Net(nn.Module):\\n    def forward(self, x):\\n        return self.fc(x)"
    }

    Respond ONLY with a JSON object following this format.
    Generated code:
    """

    parser_query = code_parser_instructions + " " + response_text
    
    response = agent_parser.run(parser_query)
    print("RAW PARSER RESPONSE TEXT:\n", repr(response.content))
    response_text = response.content
    parsed_response = json.loads(response_text)

    print("PARSED RESPONSE TEXT:\n", parsed_response)

    imports_str = parsed_response.get("imports", "")
    if isinstance(imports_str, list):
        imports_str = "\n".join(imports_str)

    return CodeOutput(
        prefix=parsed_response.get("prefix", ""),
        imports=imports_str,
        code=parsed_response.get("code", "")
    )


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
    code_solution = generate_code(messages, config, state)

    

    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem:\n\n"
            f"Prefix:\n{code_solution.prefix}\n\n"
            f"Imports:\n```\n{textwrap.dedent(code_solution.imports)}\n```\n\n"
            f"Code:\n```\n{textwrap.dedent(code_solution.code)}\n```"
           
        )
    ]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(type(code_solution.code))
    
    #state.code_generation = code_solution.code
    #state.messages = messages


    # Increment
    state.code_iterations = state.code_iterations + 1
    return {"code_generation": code_solution} #, "user_feedback": state.user_feedback


import json

def extract_packages_from_imports(import_block: str):
    """
    Use LLM to extract pip-installable packages from import block as a comma-separated string.
    """
    print("Extracting packages from imports...", import_block)

    prompt_template = """
        You are given a block of Python import statements.

        Your task is to return ONLY the pip-installable package names as a single comma-separated string.

        ### Instructions:
        - Output ONLY the top-level package names, separated by commas (e.g., torch,snntorch,matplotlib).
        - Do NOT include explanations, markdown, quotes, or code fences.
        - Exclude standard library modules (e.g., sys, os, typing).
        - Only include packages that are explicitly imported (e.g., for `import numpy as np`, return `numpy`).
        - In `from x import y` statements, include only `x`.
        - Preserve the order in which packages appear.
        - Ignore relative imports and local files.
        


        ### Imports:
        {import_block}
        """
    prompt = prompt_template.format(import_block=import_block)

    agent = Agent(
        model=Ollama(id="mistral-nemo"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=False,
    )

    try:
        response = agent.run(prompt)
        print("RAW RESPONSE TEXT PACKAGES EXTRACTOR:\n", response)

        response_text = response.content if hasattr(response, "content") else str(response)

        # Remove code fences and cleanup
        if "```" in response_text:
            response_text = response_text.strip("` \n")

        # Parse into clean list
        package_list = [pkg.strip() for pkg in response_text.split(",") if pkg.strip()]
        return package_list

    except Exception as e:
        print("Fallback to empty package list due to error:", e)
        return []







def check_code_sandbox(state: SummaryState, config: RunnableConfig):
    """
    Check code in a sandbox with dependency installation.
    """


    imports = state.code_generation.imports or ""
    print("Imports to give to extract_packages:", imports)
    code = state.code_generation.code or ""
    combined_code = f"{imports}\n{code}"


    sandbox_pyright = Sandbox("OpenEvalsPython") # already with pyright and uv installed

    # Static type check
    evaluator_pyright = create_e2b_pyright_evaluator(sandbox=sandbox_pyright)
    eval_result_pyright = evaluator_pyright(outputs=combined_code)
    print("Pyright result:", eval_result_pyright)

    
    sandbox_execution = Sandbox() # with this use sbx.run_code

    # SandBox metrics are in a private beta for now
    # metrics = sandbox.get_metrics() 
    # print("Sandbox metrics:", metrics)

    

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
        result = sandbox_execution.commands.run(f"pip install {pkg}", timeout=0)
        print(result.stdout or result.stderr)

   

    # Execution check
    evaluator_exec = create_e2b_execution_evaluator(sandbox=sandbox_execution)
    eval_result_execution = evaluator_exec(outputs=combined_code)
    print("Execution result:", eval_result_execution)

    

    return {
        "sandbox_feedback_pyright": eval_result_pyright,
        "sandbox_feedback_execution": eval_result_execution
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
        model=Ollama(id="mistral-nemo"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    # Convert dictionaries to strings
    pyright_feedback_str = json.dumps(pyright_feedback, indent=2)
    execution_feedback_str = json.dumps(execution_feedback, indent=2)

    # Construct the query using f-strings for better readability
    query = code_reflection_instructions + "\n" + pyright_feedback_str + "\n" + execution_feedback_str
    response = agent.run(query)

    # Get the string content from RunResponse
    response_content = response.content if hasattr(response, 'content') else str(response)
    print("REFLECTION RESPONSE:", response_content)


    # Store response in a variable
    code_reflection = response_content

    code_reflection = "Few checks on the generated code" + code_reflection

    

    return {
        "code_reflection": code_reflection,
        
    }


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


def collect_feedback(state: SummaryState, config: RunnableConfig):
    print("---COLLECTING FEEDBACK FROM USER---")

    feedback_prompt = {
        "instruction": "Please review the code and choose: approve, regenerate. Then explain your decision.",
        "code_review": {
            "prefix": state.code_generation.prefix,
            "imports": state.code_generation.imports,
            "code": state.code_generation.code,
        }
    }

    interrupt(feedback_prompt)





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

    ### Definitions:
    - Use **"evaluation"** if the user wants to perform an evaluation or if the user talks about to evaluate.
    - Use **"approve"** if the user is fully satisfied and wants to keep the code exactly as it is, with **no changes requested**.
    - Use **"regenerate"** if the user asks for **any modifications**, **improvements**, or expresses **dissatisfaction** with the current code.
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

        if action not in {"regenerate", "approve", "evaluation"}:
            action = "regenerate"

    except Exception as e:
        print(f"Error parsing feedback response: {e}")
        action = "regenerate"

    state.user_feedback_processed = action
    return {"user_feedback_processed": action}



def collect_feedback_evaluation(state: SummaryState, config: RunnableConfig):
    print("---COLLECTING FEEDBACK FOR EVALUATION FROM USER---")

    feedback_prompt = {
        "instruction": "Please insert the code to compare the generated code with.",
        "code_review": {
            "prefix": state.code_generation.prefix,
            "imports": state.code_generation.imports,
            "code": state.code_generation.code,
        }
    }

    interrupt(feedback_prompt)





def code_normalization(state: SummaryState, config: RunnableConfig):
    """Process the user feedback, produce the codes to compare with the same inputs"""
    import json

    print("---NORMALIZING GENERATED CODE AND REFERENCE CODE---")
    print("\nstate in process feedback:", state)

    agent = Agent(
        model=Ollama(id="deepseek-r1"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    prompt = f"""
You are a code normalization assistant.

You are given two Python code snippets:

<<GENERATED>>
{state.code_generation.code}
<<END_GENERATED>>

<<REFERENCE>>
{state.research_topic}
<<END_REFERENCE>>

Task:

- Produce ONE self-contained Python script as output.
- The script must contain two clear sections:
    ### Normalized generated
    <python code adapted from generated snippet>

    ### Normalized reference
    <python code adapted from reference snippet>
- Both sections MUST:
    * Define and use the SAME constant TOY_INPUT = <toy_input>.
    * If one code cannot directly accept TOY_INPUT, ADAPT it (e.g., by adding conversions or reshaping) so both work with exactly the same TOY_INPUT.
- IMPORTANT: Return ONLY the Python script text.
- At the end of the script:
    * Add code that runs both normalized functions on TOY_INPUT.
    * Print both results clearly, in the format:
      Generated output: <value>
      Reference output: <value>
  No JSON wrapping, no markdown fences (```), no extra commentary.
  """

    script_str = ""

    try:
        response = agent.run(prompt)
        script_str = getattr(response, "content", str(response))

        # Remove <think> artifacts
        while "<think>" in script_str and "</think>" in script_str:
            start = script_str.find("<think>")
            end = script_str.find("</think>") + len("</think>")
            script_str = script_str[:start] + script_str[end:]

        # Remove ``` fences if the model added them
        script_str = script_str.strip()
        if script_str.startswith("```"):
            script_str = script_str.strip("`")
            # sometimes comes like ```python\n...\n```
            script_str = script_str.replace("python\n", "", 1).strip()

    except Exception as e:
        print("Error while running normalization agent:", e)
        print(traceback.format_exc())
        script_str = ""

    return {"code": script_str}





def search_relevant_sources(state: SummaryState, config: RunnableConfig):
    """
    Search for relevant sources based on the user's request
    """
    print("---SEARCHING RELEVANT SOURCES TO CODE---")

    max_results = 2 # number of results to return
    include_raw_content = True

    # Customize config to not emit tool calls
    config = copilotkit_customize_config(config, emit_tool_calls=False)
    
    #await copilotkit_emit_message(config, json.dumps({
    #    "node": "search_relevant_sources",
    #    "content": f"Searching for: {state.search_query}"
    #}))
    
    configurable = Configuration.from_runnable_config(config)
    # Search the web
    agent = Agent(
        model=Ollama(id="mistral-nemo"),
        tools=[GoogleSearchTools()],
        show_tool_calls=False,
        markdown=True
        
    )

    query = code_search_instructions + "\n Search for the following query: " + state.research_topic + "\n"
    #run_response = agent.run(query)
    run_response = agent.run(query)  

    content = run_response.content

    print(f"Search results: {content}")

    return {"urls": content}


def evaluation(state: SummaryState, config: RunnableConfig):
    """
    Perform a systematic evaluation with ground truth code in an E2B sandbox.
    """
    print("---PERFORMING EVALUATION---")

    code_output = state.code_generation
    imports = code_output.imports
    code = code_output.code

    # Extract packages from import lines
    import_lines = imports.split("\n")
    packages = set()
    for line in import_lines:
        line = line.strip()
        if line.startswith("import "):
            parts = line.split()
            if len(parts) >= 2:
                packages.add(parts[1].split(".")[0])
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) >= 2:
                packages.add(parts[1].split(".")[0])

    # Convert to pip install command
    pip_cmd = "pip install " + " ".join(sorted(packages)) if packages else ""

    # Combine everything
    executable_code = f"""
    import subprocess
    import sys

    # Install necessary packages
    subprocess.run([sys.executable, "-m", "pip", "install", {" ".join(repr(pkg) for pkg in packages)}])

    # --- Imports ---
    {imports}

    # --- Code ---
    {code}
        """.strip()

    # Now run the code in E2B sandbox


    sandbox = Sandbox("OpenEvalsPython")

    # Optional: run pip install directly (if using SDK-level install)
    # await sandbox.run(pip_cmd)

    evaluator = create_e2b_execution_evaluator(sandbox=sandbox)

    result = evaluator(outputs=executable_code)

    print("EVALUATION RESULT:")
    print(result)

    return {"evaluation_result": result}



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
builder.add_node("reflection", reflection)  # reflect on code
builder.add_node("evaluation", evaluation) # perform a systematic evaluation with groud truth code

builder.add_node("collect_feedback_evaluation", collect_feedback_evaluation)  # collect feedback for evaluation
builder.add_node("code_normalization", code_normalization)  # process feedback for evaluation

# Add edges
builder.add_edge(START, "route_question")


builder.add_edge("generate_query", "web_research")
builder.add_conditional_edges(
    "route_question",
    lambda state: state.route,  # Read route from state
    {
        "Academic Source": "generate_academic_query",
        "General Web Search": "generate_query",
        #"Code": "generate",
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
    },
)

builder.add_edge("collect_feedback_evaluation", "code_normalization")
#builder.add_edge("evaluation", END)
builder.add_edge("code_normalization", END)




# Add memory saver for checkpointing
memory = MemorySaver()

# Compile graph with checkpointing and interrupts
graph = builder.compile(
#checkpointer=memory,
#interrupt_after=[""],
)

