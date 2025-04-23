"use client";

import React from "react";
import {
  AssistantMessageProps,
  CopilotChat,
  ResponseButtonProps,
} from "@copilotkit/react-ui";
import { useCoAgent, useCopilot } from "@copilotkit/react-core";
import { Brain } from "lucide-react";
import { useState } from "react";

export interface AgentState {
  research_topic: string;
  search_query: string;
  web_research_results: string[];
  sources_gathered: string[];
  research_loop_count: number;
  running_summary: string;
  user_feedback: string;
}

const ResponseButton = (props: ResponseButtonProps) => {
  return <></>;
};

const safelyParseJSON = (json: string) => {
  try {
    return JSON.parse(json);
  } catch (e) {
    return json;
  }
};

const AssistantMessage = (props: AssistantMessageProps) => {
  const { sendUserMessage } = useCopilot();
  const [feedbackSent, setFeedbackSent] = useState(false);

  if (props.message) {
    const parsed = safelyParseJSON(props.message);

    // 💡 Handle interrupt() prompt (human-in-the-loop)
    if (
      typeof parsed === "object" &&
      parsed.question &&
      (parsed.code || parsed.prefix || parsed.imports)
    ) {
      return (
        <div className="p-4 bg-white shadow-md rounded space-y-2">
          <div className="font-semibold text-sm text-gray-700">
            {parsed.question}
          </div>

          {parsed.prefix && (
            <div>
              <strong className="text-xs text-gray-600">Prefix:</strong>
              <pre className="bg-gray-100 text-xs p-2 rounded text-gray-800 whitespace-pre-wrap">
                {parsed.prefix}
              </pre>
            </div>
          )}

          {parsed.imports && (
            <div>
              <strong className="text-xs text-gray-600">Imports:</strong>
              <pre className="bg-gray-100 text-xs p-2 rounded text-gray-800 whitespace-pre-wrap">
                {parsed.imports}
              </pre>
            </div>
          )}

          {parsed.code && (
            <div>
              <strong className="text-xs text-gray-600">Code:</strong>
              <pre className="bg-gray-100 text-xs p-2 rounded text-gray-800 whitespace-pre-wrap">
                {parsed.code}
              </pre>
            </div>
          )}

          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const form = e.currentTarget;
              const feedback = new FormData(form).get("feedback");

              if (feedback) {
                console.log("SENDING FEEDBACK:", feedback.toString());
                await sendUserMessage({
                  type: "Command",
                  name: "user_feedback", // or whatever name your LangGraph agent expects
                  input: feedback.toString(), // 👈 actual user feedback
                });
                form.reset();
                setFeedbackSent(true);
                setTimeout(() => setFeedbackSent(false), 2000);
              }
            }}
            className="space-y-2 mt-2"
          >
            <textarea
              name="feedback"
              required
              className="w-full p-2 text-sm border rounded"
              rows={3}
              placeholder="Type your feedback here..."
            />
            <button
              type="submit"
              className="px-3 py-1 bg-blue-500 text-white text-sm rounded"
            >
              Submit Feedback
            </button>
            {feedbackSent && (
              <p className="text-green-600 text-xs">Feedback sent!</p>
            )}
          </form>
        </div>
      );
    }

    // 🌐 Handle node/content streaming messages
    if (typeof parsed === "object" && parsed.node && parsed.content) {
      if (
        parsed.node === "Finalize Summary" &&
        parsed.content !== "Finalizing research summary..."
      ) {
        return null;
      }

      return (
        <div className="flex flex-col gap-2 p-3 rounded bg-gray-50/30">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Brain className="w-4 h-4" />
            <span className="font-mono">{parsed.node}</span>
          </div>
          <div className="text-sm font-medium pl-6 text-gray-800">
            {parsed.content}
          </div>
        </div>
      );
    }

    // 🔎 Handle fallback parsed values
    return (
      <div className="text-sm">
        {typeof parsed === "string" ? (
          <div className="text-gray-800">{parsed}</div>
        ) : parsed.knowledge_gap ? null : (
          <div className="text-gray-800">{JSON.stringify(parsed)}</div>
        )}
      </div>
    );
  }

  return null;
};

export default function ChatSidebar() {
  const { state, setState, start } = useCoAgent<AgentState>({
    name: "ollama_deep_researcher",
    initialState: {
      research_topic: null,
      search_query: null,
      web_research_results: [],
      sources_gathered: [],
      research_loop_count: 0,
      running_summary: null,
    },
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1">
        <div className="h-full flex flex-col justify-end">
          <CopilotChat
            className="col-span-1"
            ResponseButton={ResponseButton}
            onSubmitMessage={async (message) => {
              setState({
                ...state,
                research_topic: message,
              });
              await start(); // only starts the agent — interrupts are resumed via sendUserMessage
            }}
            AssistantMessage={AssistantMessage}
          />
        </div>
      </div>
    </div>
  );
}
