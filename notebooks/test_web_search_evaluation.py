from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.models import OllamaModel

# Instantiate your Ollama model ONCE and use everywhere
model = OllamaModel(model="deepseek-r1:latest", base_url="http://localhost:11434")

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True
)
contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=model
)
contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=model
)
# Define generation metrics (for your summarize_sources node) WITH MODEL
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)
faithfulness = FaithfulnessMetric(threshold=0.7, model=model)

def test_web_search_retrieval():
    """Test the retrieval quality of your web_research node"""
    retrieval_context = [
        "Web search result 1: Relevant information about the query topic",
        "Web search result 2: More relevant details",
        "Web search result 3: Additional context"
    ]
    test_case = LLMTestCase(
        input="Your search query here",
        actual_output="Summary of web search results",
        retrieval_context=retrieval_context,
        expected_output="What you expect the summary to contain"
    )
    assert_test(test_case, [
        contextual_relevancy,
        contextual_precision,
        contextual_recall
    ])

def test_web_search_generation():
    """Test the summarization quality of your summarize_sources node"""
    retrieval_context = [
        "Retrieved web content snippet 1",
        "Retrieved web content snippet 2"
    ]
    test_case = LLMTestCase(
        input="User query",
        actual_output="Your LLM's generated summary from web search results",
        retrieval_context=retrieval_context
    )
    assert_test(test_case, [
        answer_relevancy,
        faithfulness
    ])

if __name__ == "__main__":
    test_web_search_retrieval()
    test_web_search_generation()
    print("✅ Web search evaluation completed!")
