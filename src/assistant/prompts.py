query_writer_instructions="""Your goal is to generate targeted web search query.

The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string",
    "aspect": "string",
    "rationale": "string"
}}
"""

router_instructions="""You are an intelligent decision-making assistant.
Your task is to analyze the user's question and determine the best source to search for a high-quality answer.
Choose between:

"Code": If the question is about programming, code snippets, or technical implementations that require specific coding knowledge or examples (e.g., Python, JavaScript, etc.).

"Academic Source": If the question requires in-depth technical, scientific, scholarly, or programming-related content typically found in academic papers, code notebooks, or specialized documentation (e.g., Google Colab, arXiv, or Stack Overflow).

"General Web Search": If the question relates to general knowledge, recent news, opinions, product info, how-tos, entertainment, or any casual or non-technical topic.

Your output must be only one of the following:

{ response : Academic Source }

{ response : General Web Search}

{ response : Code }

Analyze the user's question and reply with the best option."""



summarizer_instructions="""Your goal is to generate a high-quality summary of the web search results.

When EXTENDING an existing summary:
1. Seamlessly integrate new information without repeating what's already covered
2. Maintain consistency with the existing content's style and depth
3. Only add new, non-redundant information
4. Ensure smooth transitions between existing and new content

When creating a NEW summary:
1. Highlight the most relevant information from each source
2. Provide a concise overview of the key points related to the report topic
3. Emphasize significant findings or insights
4. Ensure a coherent flow of information

CRITICAL REQUIREMENTS:
- Start IMMEDIATELY with the summary content - no introductions or meta-commentary
- DO NOT include ANY of the following:
  * Phrases about your thought process ("Let me start by...", "I should...", "I'll...")
  * Explanations of what you're going to do
  * Statements about understanding or analyzing the sources
  * Mentions of summary extension or integration
- Focus ONLY on factual, objective information
- Maintain a consistent technical depth
- Avoid redundancy and repetition
- DO NOT use phrases like "based on the new results" or "according to additional sources"
- DO NOT add a References or Works Cited section
- DO NOT use any XML-style tags like <think> or <answer>
- Begin directly with the summary text without any tags, prefixes, or meta-commentary
"""

academic_summarizer_instructions = """Your goal is to generate a high-quality summary of the academic search results.

When creating a NEW summary:
1. Highlight the most relevant information from each source
2. Provide a concise overview of the key points related to the report topic
3. Emphasize significant findings or insights
4. Ensure a coherent flow of information

CRITICAL REQUIREMENTS:
- Start IMMEDIATELY with the summary content - no introductions or meta-commentary
- DO NOT include ANY of the following:
  * Phrases about your thought process ("Let me start by...", "I should...", "I'll...")
  * Explanations of what you're going to do
  * Statements about understanding or analyzing the sources
  * Mentions of summary extension or integration
- Focus ONLY on factual, objective information
- Maintain a consistent technical depth
- Avoid redundancy and repetition
- DO NOT use phrases like "based on the new results" or "according to additional sources"
- DO NOT add a References or Works Cited section
- DO NOT use any XML-style tags like <think> or <answer>
- Begin directly with the summary text without any tags, prefixes, or meta-commentary
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

Your tasks:
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
4. Be concise and synthetic in the question formulation

Ensure the follow-up question is self-contained and includes necessary context for web search.

Return your analysis as a JSON object:
{{ 
    "knowledge_gap": "string",
    "follow_up_query": "string"
}}"""


web_search_description = """You are a distinguished AI research with expertise
        in analyzing and synthesizing complex information. Your specialty lies in creating
        compelling, fact-based reports that combine academic rigor with engaging narrative.

        Your writing style is:
        - the output must be formatted in JSON format
        - Clear and authoritative
        - Engaging but professional
        - Fact-focused with proper citations
        - Accessible to educated non-specialists
    
    """

web_search_instructions ="""\
        You are a usefull agent that given a query and a tool for searching the web, 
        you call it accordingly in order to found the best matching resources relevant to the query.
        Doing that, you  must respect the following instructions:
        - You must separate each source in a structured way. 
        - for each source you need to append the title of the page, the url of the website from where the information was extracted,
        and the summary of the most relevant information found in the page.

    """

                
web_search_expected_output = """ \
You are a formatting assistant that heps the user formatting the incoming content,
from unstructured and unformatted text to a structured and formatted JSON object.
The incoming text is a list of web search results, for each results there will be a title, an source url and a content.
You MUST convert the incoming text into a single JSON, that contains a list of "results" with the following structure:
{{
    "results": [[
        {{"title": "string", 
          "url": "string", 
          "content": "string"}},
        {{"title": "string", 
          "url": "string", 
          "content": "string"}},
          ...
    ]]
}}
- do not return more than one JSON object.
- more than one results, needs to be in al list of "results" inside the main json object
- do not add any special or normal caracthers outside the brackets of the json object
- the main part of the response must be attached inside "content" key of the json object
- supplementary information must be into the json object as well, like the title of the page, the url of the website from where the information was extracted, and the raw content of the page
- DO NOT IN ANY WHAY INCLUDE IN THE RESPONSE ANY INFORMATION NOT FORMATTED AS DESCRIBED ABOVE IN JSON FORMAT"""


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

code_reflection_instructions = """\
    You are a code reflection agent. Your task is to reflect on the results of the checker and give the user suggestions to fix them.
    Also, you ask to the user how you can be useful.
    The results of the checker are the following:
    """

code_search_instructions = """\
    You are a code search assistant. Your task is to search for the most relevant URLS of pages that contain code snippets and examples relevant to the following request.
    Your output should be a JSON object with the following fields:
    {
        "urls": ["URL1", "URL2", ...]
    }
    """