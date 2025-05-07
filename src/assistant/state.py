import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated

from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

@dataclass(kw_only=True)
class SummaryState:

    route: str = field(default=None) # Route to the next state

    research_topic: str = field(default=None) # Report topic
    
    search_query: str = field(default=None) # Search query

    
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report
    raw_search_result: str = field(default=None) # Final report
    
    # Academic research
    academic_source_content : str = field(default=None)



    # Code generation
    code_iterations: int = field(default=0)
    max_code_iterations: int = field(default=3)
    code_generation: str = field(default=None)
    error: str = field(default=None)
    messages: Annotated[list[AnyMessage], add_messages]
    user_feedback: str = field(default=None) # to store user feedback
    user_feedback_processed : str = field(default=None) # to store user feedback after llm processes it
    
    sandbox_feedback_pyright: str = field(default=None)
    sandbox_feedback_execution: str = field(default=None)



@dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    running_summary: str = field(default=None) # Final report