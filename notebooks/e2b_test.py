#export E2B_API_KEY="e2b_4590f83de28112a9bd68eb920f368104fff15706"

from e2b_code_interpreter import Sandbox

# E2B template with uv and pyright preinstalled
sandbox = Sandbox("OpenEvalsPython")


from openevals.code.e2b.pyright import create_e2b_pyright_evaluator

evaluator = create_e2b_pyright_evaluator(
    sandbox=sandbox,
)

CODE = """
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.compile()

builder.invoke({})
"""

#eval_result = evaluator(outputs=CODE)

#print(eval_result)


from openevals.code.e2b.execution import create_e2b_execution_evaluator

evaluator = create_e2b_execution_evaluator(
    sandbox=sandbox,
)

CODE = """
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.compile()

builder.invoke({})
"""

eval_result = evaluator(outputs=CODE)

print(eval_result)