from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from bs4 import BeautifulSoup
from agno.models.ollama import Ollama
from agno.agent import Agent

# -------------------------
# Custom Embedding Wrapper
# -------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = None):
        import torch, os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def embed_documents(self, texts):
        import torch, gc
        batch_size = 8 if self.model.device == "cuda" else 2
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


# -------------------------
# 1. Crawl Website
# -------------------------
loader = RecursiveUrlLoader(
    url="https://snntorch.readthedocs.io/en/latest/",  # 👈 base URL
    max_depth=4,
    extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
)
docs = loader.load()
print(f"Loaded {len(docs)} documents from site")

# -------------------------
# 2. Split & Embed
# -------------------------
doc_splits = load_and_split(docs, chunk_size=512, overlap=50)
embedding_model = SentenceTransformerEmbeddings("mchochlov/codebert-base-cd-ft")

# -------------------------
# 3. Vectorstore + Retriever
# -------------------------
vectorstore = build_vectorstore(doc_splits, embedding_model, collection="snntorch-docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------
# 4. Agent with Ollama
# -------------------------
agent = Agent(
    model=Ollama(id="gpt-oss:20b"),
    tools=[],
    show_tool_calls=False,
    use_json_mode=True,
)

# -------------------------
# 5. Ask Questions
# -------------------------
query = """
Create a step-by-step tutorial with example code using snntorch.

1. Use a synthetic multivariate temporal dataset.
   The dataset shape should be: (batch_size=32, time_steps=50, features=20).

2. Clearly explain the shape of the data and how it is converted into spike trains for input into the SNN.

3. Build an SNN model with the following specifications:
   - Input layer: 20 neurons
   - Hidden layer: 100 LIF (Leaky Integrate-and-Fire) neurons
   - Output layer: 2 classes (binary classification)

4. Show the full code for defining the model, the loss function, and a basic training loop.

The tutorial should be clear and detailed, suitable for someone learning how to build an SNN on non-MNIST data.
"""
docs = retriever.get_relevant_documents(query)

# Inject retrieved context into the agent
context = "\n\n".join([d.page_content for d in docs])
prompt = f"""You are an expert on snntorch.
Use the following documentation context to answer the question.

Context:
{context}

Question:
{query}
"""

response = agent.run(prompt)
content=response.content
print("\nAnswer:\n", content)
