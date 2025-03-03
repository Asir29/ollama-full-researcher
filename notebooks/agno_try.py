from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
from assistant.configuration import Configuration

agent = Agent(
        model=Ollama(id="llama3.1:8b"),
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
    )


agent.print_response("What is the capital of France?", markdown=True)