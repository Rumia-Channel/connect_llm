use super::protocol::{
    GitHubCopilotFunctionDefinition, GitHubCopilotMessage, GitHubCopilotRequest,
    GitHubCopilotResponse, GitHubCopilotToolCall, GitHubCopilotToolDefinition,
    GitHubCopilotToolFunction, parse_tool_calls,
};
use crate::ai::{
    ChatRequest, ChatResponse, DebugTrace, Message, ThinkingEffort, ThinkingOutput, ToolCall,
    ToolChoice, ToolDefinition, Usage, providers, serialize_tool_arguments,
};
use serde_json::{Value, json};

pub(super) fn normalized_base_url(base_url: &str) -> &str {
    base_url.trim_end_matches('/')
}

pub(super) fn chat_completions_url(base_url: &str) -> String {
    let base_url = normalized_base_url(base_url);
    format!("{}/chat/completions", base_url)
}

pub(super) fn models_url(base_url: &str) -> String {
    let base_url = normalized_base_url(base_url);
    format!("{}/models", base_url)
}

fn convert_effort(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Minimal => "minimal",
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::XHigh => "xhigh",
    }
}

pub(super) fn initiator_for_messages(messages: &[GitHubCopilotMessage]) -> &'static str {
    if messages
        .last()
        .map(|message| message.role.as_str() != "user")
        .unwrap_or(false)
    {
        "agent"
    } else {
        "user"
    }
}

fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<GitHubCopilotToolDefinition>> {
    if tools.is_empty() {
        return None;
    }

    Some(
        tools
            .iter()
            .map(|tool| GitHubCopilotToolDefinition {
                tool_type: "function",
                function: GitHubCopilotFunctionDefinition {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.input_schema.clone(),
                },
            })
            .collect(),
    )
}

fn convert_tool_choice(choice: Option<&ToolChoice>) -> Option<Value> {
    match choice? {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Tool(name) => Some(json!({
            "type": "function",
            "function": {
                "name": name,
            }
        })),
    }
}

fn convert_tool_calls(tool_calls: Vec<ToolCall>) -> Option<Vec<GitHubCopilotToolCall>> {
    if tool_calls.is_empty() {
        return None;
    }

    Some(
        tool_calls
            .into_iter()
            .map(|tool_call| GitHubCopilotToolCall {
                id: tool_call.id,
                call_type: "function".to_string(),
                function: GitHubCopilotToolFunction {
                    name: tool_call.name,
                    arguments: serialize_tool_arguments(&tool_call.arguments),
                },
            })
            .collect(),
    )
}

pub(super) fn convert_request(request: ChatRequest, stream: bool) -> GitHubCopilotRequest {
    let ChatRequest {
        model,
        messages: request_messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        system,
        thinking,
    } = request;
    let request_policy = providers::github_copilot::spec().request_policy(&model);
    let temperature = request_policy.sanitize_temperature(temperature);

    let mut messages = Vec::new();

    if let Some(system) = system {
        messages.push(GitHubCopilotMessage {
            role: "system".to_string(),
            content: Some(system),
            reasoning_text: None,
            reasoning_opaque: None,
            tool_call_id: None,
            tool_calls: None,
        });
    }

    for message in request_messages {
        let Message {
            role,
            content,
            thinking,
            tool_calls,
            tool_call_id,
            tool_name: _,
            tool_result: _,
            tool_error: _,
        } = message;

        let (reasoning_text, reasoning_opaque) = match thinking {
            Some(thinking) => (thinking.text, thinking.signature.or(thinking.redacted)),
            None => (None, None),
        };

        let content = if role == "assistant" && content.is_empty() {
            None
        } else {
            Some(content)
        };

        messages.push(GitHubCopilotMessage {
            role,
            content,
            reasoning_text,
            reasoning_opaque,
            tool_call_id,
            tool_calls: convert_tool_calls(tool_calls),
        });
    }

    let reasoning_effort = thinking
        .as_ref()
        .and_then(|thinking| thinking.effort)
        .map(convert_effort);
    let thinking_budget = thinking
        .as_ref()
        .and_then(|thinking| thinking.enabled.then_some(thinking.budget_tokens))
        .flatten();
    let reasoning_effort = request_policy.sanitize_reasoning_effort(reasoning_effort);
    let thinking_budget = request_policy.sanitize_thinking_budget(thinking_budget);

    GitHubCopilotRequest {
        model,
        messages,
        tools: convert_tools(&tools),
        tool_choice: convert_tool_choice(tool_choice.as_ref()),
        max_tokens,
        temperature,
        reasoning_effort,
        thinking_budget,
        stream,
    }
}

pub(super) fn convert_response(
    response: GitHubCopilotResponse,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let message = response.choices.first().map(|choice| &choice.message);
    let content = message
        .and_then(|message| message.content.clone())
        .unwrap_or_default();
    let thinking = message.and_then(|message| {
        let thinking = ThinkingOutput {
            text: message.reasoning_text.clone(),
            signature: message.reasoning_opaque.clone(),
            redacted: None,
        };
        (!thinking.is_empty()).then_some(thinking)
    });
    let tool_calls = message
        .map(|message| parse_tool_calls(message.tool_calls.clone()))
        .unwrap_or_default();
    let usage = response.usage.unwrap_or_default();

    ChatResponse {
        id: response.id,
        content,
        model: response.model,
        usage: Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
        thinking,
        tool_calls,
        debug: if request_debug.is_some() || response_debug.is_some() {
            Some(DebugTrace {
                request: request_debug,
                response: response_debug,
            })
        } else {
            None
        },
    }
}
