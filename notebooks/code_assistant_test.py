from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


llm = ChatOllama(model="llama2", temperature=0.1)

code_gen_prompt_claude = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables \n
            defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
            \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    #description: ClassVar[str] = "Schema for code solutions to questions about LCEL."

code_gen_chain = llm.with_structured_output(code, include_raw=False)


question = "Write a function for fibonacci."
messages = [("user", question)]

# Test
result = code_gen_chain.invoke(messages)
print(result)