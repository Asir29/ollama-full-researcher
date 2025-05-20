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

code_assistant_instructions = """ \
    You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables 
    defined. Structure your answer in JSON format with the following fields:
    {
        "prefix": "A description of the code solution",
        "imports": "The necessary import statements",
        "code": "The functioning code block"
    }
    You MUST avoid to include any explanation or meta-commentary.
    You MUST avoid to include any new lines or special characters outside the brackets of the JSON object.
    The response MUST contain only the fields specified in the JSON format.
    DO NOT include Markdown formatting (no ``` or code blocks).

    
    

    """

class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


# Proper embedding wrapper for LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def generate_code(question: str, urls: str) -> CodeOutput:
    agent = Agent(
        model=Ollama(id="codellama"),
        tools=[],
        show_tool_calls=False,
        use_json_mode=True,
    )

    # Load URLs
    data = json.loads(urls)
    urls = data["urls"]
    print(f"URLs: {urls}")

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [doc for sublist in docs for doc in sublist]
    print(f"Loaded {len(docs_list)} documents")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Use lightweight embedding model
    embedding_model = SentenceTransformerEmbeddings("nomic-ai/CodeRankEmbed")

    # Vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        collection_name="code-rag",
    )

    # Retrieve
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("CONTEXT:", context)

    # Construct query
    query = code_assistant_instructions + "\n Based on the following context, generate the code that satisfies the question:" + "Context: " + context + "\nQuestion: " + question

    # Run agent
    response = agent.run(query)
    response_text = response.content
    print("RAW RESPONSE TEXT:\n", repr(response_text))

    # Parse result
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error during JSON parsing: {e}")
        return CodeOutput(prefix="", imports="", code="")

    imports_str = parsed_response.get("imports", "")
    if isinstance(imports_str, list):
        imports_str = "\n".join(imports_str)

    return CodeOutput(
        prefix=parsed_response.get("prefix", ""),
        imports=imports_str,
        code=parsed_response.get("code", "")
    )


# Example usage
if __name__ == "__main__":
    urls = '{"urls": ["https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html"]}'
    question = "How to build a Surrogate gradient descent for spike nn?"
    code_output = generate_code(question, urls)

    print("Generated Code Output:")
    print("Prefix:", code_output.prefix)
    print("Imports:", code_output.imports)
    print("Code:", code_output.code)
