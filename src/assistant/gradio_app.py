# gradio_app.py

import gradio as gr
import asyncio
import json

from graph import graph
from state import SummaryStateInput, SummaryStateOutput
from configuration import Configuration
from langchain_core.runnables import RunnableConfig

# Global state placeholder to store interrupts or decisions
feedback_decision = {}

# Async function to run LangGraph and handle interrupts
def run_langgraph_with_feedback(prompt):
    config = RunnableConfig(
        
    )

    input_state = SummaryStateInput(
        research_topic=prompt,
        messages=[],
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    result = loop.run_until_complete(graph.ainvoke(input_state, config))

    summary = getattr(result, "running_summary", None)
    if not summary:
        summary = "No summary was generated."

    return summary

# Simulated feedback interrupt handler for demo
# In practice, you would wire this to copilotkit's real interrupt feedback
with gr.Blocks() as demo_ui:
    gr.Markdown("""
        # LangGraph AI Research Assistant
        Enter a research question or coding task below and get a summarized, AI-driven response.
    """)

    with gr.Row():
        topic_input = gr.Textbox(label="Research Topic or Code Request", placeholder="e.g., How does attention work in transformers?")

    run_button = gr.Button("Run LangGraph")
    output = gr.Textbox(label="AI Summary or Code Output")

    feedback = gr.Radio([
        "approve",
        "regenerate",
        "evaluation"
    ], label="Feedback (if asked by the agent)", visible=False)
    send_feedback = gr.Button("Send Feedback", visible=False)

    def launch_graph(prompt):
        feedback.visible = False
        send_feedback.visible = False
        summary = run_langgraph_with_feedback(prompt)
        # Optionally display feedback options if summary includes a prompt for review
        if "Please review the code" in summary or "approve, regenerate" in summary:
            feedback.visible = True
            send_feedback.visible = True
        return summary, gr.update(visible=feedback.visible), gr.update(visible=send_feedback.visible)

    def save_feedback(choice):
        feedback_decision["choice"] = choice
        return f"Your feedback '{choice}' has been recorded."

    run_button.click(fn=launch_graph, inputs=topic_input, outputs=[output, feedback, send_feedback])
    send_feedback.click(fn=save_feedback, inputs=feedback, outputs=output)

if __name__ == "__main__":
    demo_ui.launch()
