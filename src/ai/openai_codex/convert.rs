use super::protocol::{
    OpenAiCodexFunctionCallItem, OpenAiCodexFunctionCallOutputItem, OpenAiCodexInputContent,
    OpenAiCodexInputItem, OpenAiCodexInputMessage, OpenAiCodexOutputItem,
    OpenAiCodexReasoningRequest, OpenAiCodexRequest, OpenAiCodexResponse, OpenAiCodexTool,
    OpenAiCodexToolChoice,
};
use crate::ai::{
    ChatRequest, ChatResponse, DebugTrace, Message, ThinkingConfig, ThinkingEffort, ThinkingOutput,
    ToolCall, ToolChoice, Usage, parse_tool_arguments, providers, serialize_tool_arguments,
};

const DEFAULT_CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";

pub(super) fn endpoint_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.is_empty() {
        return DEFAULT_CODEX_ENDPOINT.to_string();
    }
    if trimmed.ends_with("/codex/responses") {
        trimmed.to_string()
    } else {
        format!("{}/codex/responses", trimmed)
    }
}

fn convert_reasoning_config(
    thinking: Option<&ThinkingConfig>,
) -> Option<OpenAiCodexReasoningRequest> {
    let thinking = thinking?;

    if !thinking.enabled {
        return Some(OpenAiCodexReasoningRequest {
            effort: Some("none"),
            summary: None,
        });
    }

    let effort = match thinking.effort.unwrap_or(ThinkingEffort::Medium) {
        ThinkingEffort::Minimal => "minimal",
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::XHigh => "xhigh",
    };

    Some(OpenAiCodexReasoningRequest {
        effort: Some(effort),
        summary: Some("auto"),
    })
}

fn default_instructions() -> String {
    "You are a helpful assistant.".to_string()
}

fn convert_tools(tools: Vec<crate::ai::ToolDefinition>) -> Option<Vec<OpenAiCodexTool>> {
    if tools.is_empty() {
        return None;
    }

    Some(
        tools
            .into_iter()
            .map(|tool| OpenAiCodexTool {
                tool_type: "function",
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema,
            })
            .collect(),
    )
}

fn convert_tool_choice(tool_choice: Option<ToolChoice>) -> Option<OpenAiCodexToolChoice> {
    match tool_choice {
        None => None,
        Some(ToolChoice::Auto) => Some(OpenAiCodexToolChoice::Mode("auto")),
        Some(ToolChoice::None) => Some(OpenAiCodexToolChoice::Mode("none")),
        Some(ToolChoice::Required) => Some(OpenAiCodexToolChoice::Mode("required")),
        Some(ToolChoice::Tool(name)) => Some(OpenAiCodexToolChoice::Function {
            choice_type: "function",
            name,
        }),
    }
}

pub(super) fn convert_request(request: ChatRequest, stream: bool) -> OpenAiCodexRequest {
    let ChatRequest {
        model,
        messages: request_messages,
        tools,
        tool_choice,
        max_tokens: _,
        temperature,
        system,
        thinking,
    } = request;
    let request_policy = providers::openai_codex::spec().request_policy(&model);
    let temperature = request_policy.sanitize_temperature(temperature);

    let mut input = Vec::new();

    for message in request_messages {
        match message {
            Message::Tool {
                tool_call_id,
                result,
                ..
            } => input.push(OpenAiCodexInputItem::FunctionCallOutput(
                OpenAiCodexFunctionCallOutputItem {
                    item_type: "function_call_output",
                    call_id: tool_call_id,
                    output: serialize_tool_arguments(&result),
                },
            )),
            Message::User {
                content,
                created_at_ms: _,
            } => {
                if !content.is_empty() {
                    input.push(OpenAiCodexInputItem::Message(OpenAiCodexInputMessage {
                        role: "user".to_string(),
                        content: vec![OpenAiCodexInputContent {
                            content_type: "input_text",
                            text: content,
                        }],
                    }));
                }
            }
            Message::Assistant {
                content,
                created_at_ms: _,
                thinking: _,
                tool_calls,
            } => {
                if !content.is_empty() {
                    input.push(OpenAiCodexInputItem::Message(OpenAiCodexInputMessage {
                        role: "assistant".to_string(),
                        content: vec![OpenAiCodexInputContent {
                            content_type: "output_text",
                            text: content,
                        }],
                    }));
                }

                for tool_call in tool_calls {
                    input.push(OpenAiCodexInputItem::FunctionCall(
                        OpenAiCodexFunctionCallItem {
                            item_type: "function_call",
                            call_id: tool_call.id,
                            name: tool_call.name,
                            arguments: serialize_tool_arguments(&tool_call.arguments),
                        },
                    ));
                }
            }
        }
    }

    OpenAiCodexRequest {
        model,
        instructions: system.unwrap_or_else(default_instructions),
        input,
        max_output_tokens: None,
        temperature,
        reasoning: convert_reasoning_config(thinking.as_ref()),
        tools: convert_tools(tools),
        tool_choice: convert_tool_choice(tool_choice),
        stream,
        store: false,
    }
}

fn extract_text_from_output(output: &[OpenAiCodexOutputItem]) -> String {
    output
        .iter()
        .filter(|item| item.item_type == "message" && item.role.as_deref() == Some("assistant"))
        .flat_map(|item| item.content.iter())
        .filter(|part| part.content_type == "output_text")
        .filter_map(|part| part.text.clone())
        .collect::<Vec<_>>()
        .join("")
}

fn extract_thinking_from_output(output: &[OpenAiCodexOutputItem]) -> Option<ThinkingOutput> {
    let text = output
        .iter()
        .filter(|item| item.item_type == "reasoning")
        .flat_map(|item| item.summary.iter())
        .filter_map(|part| part.text.clone())
        .collect::<Vec<_>>()
        .join("");

    if text.is_empty() {
        None
    } else {
        Some(ThinkingOutput {
            text: Some(text),
            signature: output
                .iter()
                .find(|item| item.item_type == "reasoning")
                .and_then(|item| item.encrypted_content.clone()),
            redacted: None,
        })
    }
}

fn extract_tool_calls_from_output(output: &[OpenAiCodexOutputItem]) -> Vec<ToolCall> {
    output
        .iter()
        .filter(|item| item.item_type == "function_call")
        .filter_map(|item| {
            let id = item.call_id.clone().or_else(|| item.id.clone())?;
            let name = item.name.clone()?;
            let arguments = item.arguments.as_deref().map(parse_tool_arguments)?;
            Some(ToolCall {
                id,
                name,
                arguments,
            })
        })
        .collect()
}

pub(super) fn convert_response(
    response: OpenAiCodexResponse,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let usage = response.usage.unwrap_or_default();

    ChatResponse {
        id: response.id,
        content: extract_text_from_output(&response.output),
        model: response.model,
        usage: Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        },
        thinking: extract_thinking_from_output(&response.output),
        images: Vec::new(),
        tool_calls: extract_tool_calls_from_output(&response.output),
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
