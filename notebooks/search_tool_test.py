# from langchain_community.tools import DuckDuckGoSearchResults
# search = DuckDuckGoSearchResults()
# search.invoke("Obama")

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.baidusearch import BaiduSearchTools



code_search_instructions = """\
You are a code search assistant.

Rules:
1. You MUST use the DuckDuck tool to find relevant urls.  
   You are NOT allowed to generate or guess URLs under any circumstances.  
   The only URLs you can return must come directly from the tool output.
2. The final response MUST be a single valid JSON object in this format:
{{
    "urls": ["URL1", "URL2", ...]
}}

Perform the search now for the following query: "{research_topic}"
"""

# Optional: use a working proxy if needed
google_tool = GoogleSearchTools(proxy=None)  # or proxy="http://your_proxy:port"
duck_tool = DuckDuckGoTools(fixed_max_results=3)
baidusearch_tool = BaiduSearchTools(fixed_max_results=3)

agent = Agent(
    model=Ollama(id="qwen3:latest"),
    tools=[duck_tool],       # ✅ must pass the actual tool
    show_tool_calls=True,
    markdown=False
)


research_topic = "What are the latest advancements in spike neural networks for AI applications?"

query = code_search_instructions.format(research_topic=research_topic)# + "\n Search for the following query: " + state.research_topic + "\n"
run_response = agent.run(query)  
print("RAW SEARCH RESPONSE:\n", run_response)
content = run_response.content

# TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
while "<think>" in content and "</think>" in content:
    start = content.find("<think>")
    end = content.find("</think>") + len("</think>")
    content = content[:start] + content[end:]

print(f"Search results: {content}")