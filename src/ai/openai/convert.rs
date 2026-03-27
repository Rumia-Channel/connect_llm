use super::protocol::{
    OpenAiExtraBody, OpenAiFunctionDefinition, OpenAiGoogleExtraBody, OpenAiGoogleThinkingConfig,
    OpenAiMessage, OpenAiRequest, OpenAiResponse, OpenAiThinkingRequest, OpenAiToolCall,
    OpenAiToolDefinition, OpenAiToolFunction, convert_tool_calls_to_response,
};
use crate::ai::{
    ChatRequest, ChatResponse, DebugTrace, Message, ThinkingConfig, ThinkingOutput, ToolCall,
    ToolChoice, ToolDefinition, Usage, providers, serialize_tool_arguments,
};
use serde_json::{Value, json};

pub(super) fn normalized_base_url(base_url: &str) -> &str {
    base_url.trim_end_matches('/')
}

pub(super) fn chat_completions_url(base_url: &str) -> String {
    let base_url = normalized_base_url(base_url);
    if base_url.ends_with("/v1") || base_url.contains("/paas/v4") || base_url.ends_with("/openai") {
        format!("{}/chat/completions", base_url)
    } else {
        format!("{}/v1/chat/completions", base_url)
    }
}

pub(super) fn models_url(base_url: &str) -> String {
    let base_url = normalized_base_url(base_url);
    if base_url.ends_with("/v1") || base_url.contains("/paas/v4") || base_url.ends_with("/openai") {
        format!("{}/models", base_url)
    } else {
        format!("{}/v1/models", base_url)
    }
}

fn supports_reasoning_config(base_url: &str) -> bool {
    let base_url = normalized_base_url(base_url);
    base_url.contains("api.moonshot.ai") || base_url.contains("api.z.ai")
}

fn is_google_openai_compat(base_url: &str) -> bool {
    normalized_base_url(base_url).contains("generativelanguage.googleapis.com")
}

fn convert_thinking_config(
    base_url: &str,
    thinking: Option<&ThinkingConfig>,
) -> Option<OpenAiThinkingRequest> {
    let thinking = thinking?;
    if !supports_reasoning_config(base_url) {
        return None;
    }

    Some(OpenAiThinkingRequest {
        thinking_type: if thinking.enabled {
            "enabled"
        } else {
            "disabled"
        },
        clear_thinking: if base_url.contains("api.z.ai") {
            thinking.clear_history
        } else {
            None
        },
    })
}

fn convert_google_extra_body(
    base_url: &str,
    thinking: Option<&ThinkingConfig>,
) -> Option<OpenAiExtraBody> {
    let thinking = thinking?;
    if !thinking.enabled || !is_google_openai_compat(base_url) {
        return None;
    }

    Some(OpenAiExtraBody {
        google: OpenAiGoogleExtraBody {
            thinking_config: OpenAiGoogleThinkingConfig {
                include_thoughts: true,
                thinking_budget: thinking.budget_tokens,
            },
        },
    })
}

fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<OpenAiToolDefinition>> {
    if tools.is_empty() {
        return None;
    }

    Some(
        tools
            .iter()
            .map(|tool| OpenAiToolDefinition {
                tool_type: "function",
                function: OpenAiFunctionDefinition {
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

fn convert_tool_calls(tool_calls: Vec<ToolCall>) -> Option<Vec<OpenAiToolCall>> {
    if tool_calls.is_empty() {
        return None;
    }

    Some(
        tool_calls
            .into_iter()
            .map(|tool_call| OpenAiToolCall {
                id: tool_call.id,
                call_type: "function".to_string(),
                function: OpenAiToolFunction {
                    name: tool_call.name,
                    arguments: serialize_tool_arguments(&tool_call.arguments),
                },
            })
            .collect(),
    )
}

pub(super) fn convert_request(request: ChatRequest, base_url: &str, stream: bool) -> OpenAiRequest {
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
    let request_policy =
        providers::openai_compatible_spec_for_base_url(base_url).request_policy(&model);
    let temperature = request_policy.sanitize_temperature(temperature);

    let mut messages = Vec::new();

    if let Some(system) = system {
        messages.push(OpenAiMessage {
            role: "system".to_string(),
            content: Some(system),
            reasoning_content: None,
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
        let reasoning_content = thinking.and_then(|thinking| thinking.text);
        messages.push(OpenAiMessage {
            role,
            content: if content.is_empty() && !tool_calls.is_empty() {
                None
            } else {
                Some(content)
            },
            reasoning_content,
            tool_call_id,
            tool_calls: convert_tool_calls(tool_calls),
        });
    }

    OpenAiRequest {
        model,
        messages,
        tools: convert_tools(&tools),
        tool_choice: convert_tool_choice(tool_choice.as_ref()),
        max_tokens,
        temperature,
        thinking: convert_thinking_config(base_url, thinking.as_ref()),
        extra_body: convert_google_extra_body(base_url, thinking.as_ref()),
        stream,
    }
}

pub(super) fn convert_response(
    response: OpenAiResponse,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let tool_calls = response
        .choices
        .first()
        .map(|choice| convert_tool_calls_to_response(choice.message.tool_calls.clone()))
        .unwrap_or_default();
    let thinking = response
        .choices
        .first()
        .and_then(|choice| {
            choice
                .message
                .reasoning_content
                .clone()
                .or_else(|| choice.message.reasoning.clone())
        })
        .map(|text| ThinkingOutput {
            text: Some(text),
            signature: None,
            redacted: None,
        });

    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .unwrap_or_default();

    ChatResponse {
        id: response.id,
        content,
        model: response.model,
        usage: Usage {
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        },
        thinking,
        images: Vec::new(),
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
