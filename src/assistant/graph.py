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

from assistant.configuration import Configuration
from assistant.utils import deduplicate_and_format_sources, format_sources
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, web_search_instructions, web_search_description, web_search_expected_output, router_instructions, code_assistant_instructions, academic_summarizer_instructions
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_exit, copilotkit_customize_config

from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
from langchain_ollama import ChatOllama
from assistant.configuration import Configuration

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

    # agent = Agent(
    #     model=Ollama(id="mistral-nemo"),
    #     tools=[GoogleSearchTools()],
    #     show_tool_calls=False,
    #     structured_outputs=True

    # )   


######################################## ACADEMIC RESEARCH BRANCH ############################################

import asyncio
from semanticscholar import SemanticScholar

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

import asyncio


async def academic_research(state, config, papers_limit=3):
    """Perform academic research for the most relevant papers."""
    sch = SemanticScholar()

    try:
        # Run sync method in background thread
        results = await asyncio.to_thread(sch.search_paper, state.search_query, limit=papers_limit)
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return {"academic_source_content": []}
    
    abstracts = [
        f"{paper.title} - {paper.abstract or 'No abstract'}"
        for paper in results
    ]

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


# Function to generate code, called by the node "generate"
def generate_code(question: str, config: RunnableConfig, state: SummaryState) -> CodeOutput:
    messages = [question]

    configurable = Configuration.from_runnable_config(config)

    agent = Agent(
        model=Ollama(id="codellama"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    
    query = code_assistant_instructions + "\n Satisfy the following instructions: " + state.research_topic + "\n"
    response = agent.run(query)  


    response_text = response.content  # This is a string

    print("RAW RESPONSE TEXT:\n", repr(response_text))

    # Parse the response into dict
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error during JSON parsing: {e}")
        return CodeOutput(prefix="", imports="", code="")

    # Ensure that `code` is always a string, even if it is empty or None
    code_str = str(parsed_response.get("code", ""))
    # Ensure that `imports` is always a string, even if it is empty or None
    #imports_str = str(parsed_response.get("imports", ""))


    return CodeOutput(
        prefix=parsed_response["prefix"],
        imports="\n".join(parsed_response["imports"]) if isinstance(parsed_response["imports"], list) else parsed_response["imports"],
        code=code_str
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

    
    state.code_generation = code_solution
    state.messages = messages


    # Increment
    state.code_iterations = state.code_iterations + 1
    return {"code_generation": code_solution, "messages": messages, "code_iterations": state.code_iterations} #, "user_feedback": state.user_feedback


def code_check(state: SummaryState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    print(f"Current state: {state}")
    # State
    messages = state.messages
    code_solution = state.code_generation
    iterations = state.code_iterations

    if code_solution is None:
        logger.error("code_solution is None. Cannot check code.")
        return
    imports = code_solution.imports

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports) # Check imports, it actually executes the imports
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "code_generation": code_solution,
            "messages": messages,
            "code_iterations": state.code_iterations,
            "error": "yes",
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        print(f"CODE TO TEST: {combined_code}")
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "code_generation": code_solution,
            "messages": messages,
            "code_iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "code_generation": code_solution,
        "messages": messages,
        "code_iterations": iterations,
        "error": "no",
    }


from e2b_code_interpreter import Sandbox

from openevals.code.e2b.pyright import create_e2b_pyright_evaluator

from openevals.code.e2b.execution import create_e2b_execution_evaluator



def check_code_sandbox(state: SummaryState, config: RunnableConfig):
    """
    Check code in a sandbox
    """

    # E2B template with uv and pyright preinstalled
    sandbox = Sandbox("OpenEvalsPython")

    # Create the evaluator
    evaluator = create_e2b_pyright_evaluator(
    sandbox=sandbox,
    )

    imports = state.code_generation.imports
    code = state.code_generation.code

    # Combine imports and code
    combined_code = f"{imports}\n{code}"

    eval_result_pyright = evaluator(outputs=combined_code)

    print(eval_result_pyright)


    evaluator = create_e2b_execution_evaluator(
    sandbox=sandbox,
    )

    eval_result_execution = evaluator(outputs=combined_code)

    print(eval_result_execution)

    return { 
        "sandbox_feedback_pyright": eval_result_pyright,
        "sandbox_feedback_execution": eval_result_execution
    }


reflection_instructions = """\
    You are a code reflection agent. Your task is to reflect on the code and decide whether to finish or retry.
    You will receive the following information:
    1. The code generated.
    2. The feedbacks from the code checker.
    You have to produce a query for the code generator that makes clear what must be modified and the entire code solution must be reported.
    The output should be a JSON object with the following fields:
    {
        "response": "The errors indicate that ..., so change the following code to fix them: ..."
    }
    Consider to report the code as it is since the code generation agent will take care of modifying it.
    """

def reflection(state: SummaryState, config: RunnableConfig):
    """
    Reflect on the code and decide whether to finish or retry.
    """

    print("---REFLECTING ON CODE---")

    # Check code
    pyright_feedback = state.sandbox_feedback_pyright
    execution_feedback = state.sandbox_feedback_execution
    print("PYRIGHT FEEDBACK:", pyright_feedback)

    if(pyright_feedback['key'] == "pyright_succeeded" and execution_feedback['key'] == "execution_succeeded"):
        print("---NO ERRORS---")
        return {"route": "no_errors"}  # Return a dictionary
    
    agent = Agent(
        model=Ollama(id="mistral-nemo"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    # Construct the query using f-strings for better readability
    query = reflection_instructions + "\n" + state.code_generation.code + "\n" + pyright_feedback + "\n" + execution_feedback
    response = agent.run(query)
    print("REFLECTION RESPONSE:", response)
    # Parse the response into a dictionary
    parsed_response = json.loads(response)
    
    state.research_topic = parsed_response

    return {"route": "regenerate"}  # Return a dictionary


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
        "instruction": "Please review the code and choose: approve, regenerate, or modify. Then explain your decision.",
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
    You are a decision agent.\n
    You receive the user feedback: {state.research_topic}.\n
    Based on that, respond with one of the following exact JSON objects:\n\n
    {{"response": "approve"}}\n
    {{"response": "modify"}}\n
    {{"response": "regenerate"}}\n\n
    Only return the JSON. Do not include any other text, logs, or thoughts."""

    


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

        if action not in {"regenerate", "modify", "approve"}:
            action = "regenerate"

    except Exception as e:
        print(f"Error parsing feedback response: {e}")
        action = "regenerate"

    state.user_feedback_processed = action
    return {"user_feedback_processed": action}


def modify_code(state: SummaryState, config: RunnableConfig):
    """
    Allow the user to edit the code directly.
    """
    print("---MODIFYING CODE---")

    agent = Agent(
        model=Ollama(id="codellama"),
        tools=[],
        show_tool_calls=False,
        structured_outputs=True,
    )

    current_code = state.code_generation

    # Construct the query using f-strings for better readability
    query = f"Rewrite the code with the user feedback: {state.user_feedback}\n{current_code}"

    response = agent.run(query)

    # Parse the response into a dictionary
    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError as e:
        # Log the error instead of printing directly
        logging.error(f"Error during JSON parsing: {e}")
        return CodeOutput(prefix="", imports="", code="")

    # Extract the updated code and ensure it is a string
    updated_code = str(parsed_response.get("code", ""))

    # Update the state with the new code
    state.code_generation = updated_code

    # Return the updated code and relevant state information
    return {
        "code_generation": updated_code,
        "messages": state.messages,
        "code_iterations": state.code_iterations,
        "user_feedback": state.user_feedback,
    }



# not used
def continue_generation(state: SummaryState, config: RunnableConfig):
    """
    Continue the generation from where it left off.
    """
    print("---CONTINUING CODE GENERATION---")

    continued_code = continue_code_generation(state.code_generation.code, config, state)

    state.code_generation.code += "\n" + continued_code
    state.messages.append(("assistant", f"Continuing code generation:\n```\n{continued_code}\n```"))

    return {
        "code_generation": state.code_generation,
        "messages": state.messages
    }



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
builder.add_node("generate", generate)  # generation solution
#builder.add_node("check_code", code_check)  # check code
builder.add_node("modify_code", modify_code)
#builder.add_node("continue_generation", continue_generation)
builder.add_node("collect_feedback", collect_feedback)
builder.add_node("process_feedback", process_feedback)
builder.add_node("check_code_sandbox", check_code_sandbox)  # check code in sandbox
builder.add_node("reflection", reflection)  # reflect on code

# Add edges
builder.add_edge(START, "route_question")
builder.add_edge("generate_query", "web_research")
builder.add_conditional_edges(
    "route_question",
    lambda state: state.route,  # Read route from state
    {
        "Academic Source": "generate_academic_query",
        "General Web Search": "generate_query",
        "Code": "generate",
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

builder.add_edge("generate", "check_code_sandbox")
#builder.add_edge("generate", "collect_feedback")

builder.add_edge("check_code_sandbox", "reflection")

builder.add_conditional_edges(
    "reflection",
    lambda state: state.route,
    {
        "no_errors": "collect_feedback",
        "regenerate": "generate",
        
    },
)

builder.add_edge("modify_code", "collect_feedback")

builder.add_edge("collect_feedback", "process_feedback")

builder.add_conditional_edges(
    "process_feedback",
    lambda state: state.user_feedback_processed,
    {
        "regenerate": "generate",
        "approve": END,
    },
)





# Add memory saver for checkpointing
memory = MemorySaver()

# Compile graph with checkpointing and interrupts
graph = builder.compile(
#checkpointer=memory,
#interrupt_after=[""],
)

