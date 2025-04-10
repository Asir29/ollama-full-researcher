from langsmith import traceable

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Tavily search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

# @traceable
# def search_web(query, include_raw_content=True, max_results=3):
#     """Search the web using Agno with DuckDuckGo and ArXiv tools.
    
#     Args:
#         query (str): The search query to execute
#         include_raw_content (bool): Whether to include the raw content in results
#         max_results (int): Maximum number of results to return
        
#     Returns:
#         dict: Search response containing:
#             - results (list): List of search result dictionaries, each containing:
#                 - title (str): Title of the search result
#                 - url (str): URL of the search result
#                 - content (str): Snippet/summary of the content
#                 - raw_content (str): Full content if available
#     """
#     # Initialize agent with search tools
#     agent = Agent()
#     agent.add_tools([
#         DuckDuckGoTools(),
#         ArxivTools()
#     ])

#     # Define the search task
#     search_task = {
#         "task": f"Search for information about: {query}",
#         "instructions": [
#             "Use both DuckDuckGo and ArXiv to find relevant information",
#             "Combine and rank results by relevance",
#             f"Return top {max_results} most relevant results",
#             "For each result, provide title, URL, summary content, and full content if available"
#         ],
#         "output_format": {
#             "results": [
#                 {
#                     "title": "str",
#                     "url": "str",
#                     "content": "str",
#                     "raw_content": "str" if include_raw_content else ""
#                 }
#             ]
#         }
#     }

#     # Execute search
#     try:
#         response = agent.execute(search_task)
#         return {
#             "results": response.get("results", [])[:max_results]
#         }
#     except Exception as e:
#         print(f"Search error: {str(e)}")
#         return {"results": []}

# # Maintain compatibility with existing code
# tavily_search = search_web

# Prompt for coding assistant
from langchain.prompts import ChatPromptTemplate

def get_prompt_code_assistant():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables 
                defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
                \n Here is the user question:""",
            ),
            ("user", "{messages}"),
        ]
    )