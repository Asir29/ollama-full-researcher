from agno.agent import Agent
from agno.models.ollama import Ollama


agent = Agent(
        model=Ollama(id="gpt-oss:20b"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )  

instructions = """
    Code to build a basic spike neural network in python using numpy.
    """

query = instructions 
    
response = agent.run(query)
print("RAW PARSER RESPONSE TEXT:\n", repr(response.content))
response_text = response.content