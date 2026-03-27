use super::protocol::{
    AnthropicRequest, AnthropicRequestContentBlock, AnthropicRequestMessage, AnthropicResponse,
    AnthropicThinkingRequest, AnthropicToolChoice, AnthropicToolDefinition,
};
use crate::ai::{
    ChatRequest, ChatResponse, DebugTrace, Message, ThinkingConfig, ThinkingDisplay,
    ThinkingOutput, ToolCall, ToolChoice, ToolDefinition, Usage,
};
use serde_json::Value;

pub(super) fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<AnthropicToolDefinition>> {
    if tools.is_empty() {
        return None;
    }

    Some(
        tools
            .iter()
            .map(|tool| AnthropicToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.input_schema.clone(),
            })
            .collect(),
    )
}

pub(super) fn convert_tool_choice(choice: Option<&ToolChoice>) -> Option<AnthropicToolChoice> {
    match choice? {
        ToolChoice::Auto => Some(AnthropicToolChoice {
            choice_type: "auto",
            name: None,
        }),
        ToolChoice::None => Some(AnthropicToolChoice {
            choice_type: "none",
            name: None,
        }),
        ToolChoice::Required => Some(AnthropicToolChoice {
            choice_type: "any",
            name: None,
        }),
        ToolChoice::Tool(name) => Some(AnthropicToolChoice {
            choice_type: "tool",
            name: Some(name.clone()),
        }),
    }
}

pub(super) fn convert_request_message(message: Message) -> AnthropicRequestMessage {
    let Message {
        role,
        content,
        thinking,
        tool_calls,
        tool_call_id,
        tool_name,
        tool_result,
        tool_error,
    } = message;

    if role == "tool" {
        return AnthropicRequestMessage {
            role: "user".to_string(),
            content: vec![AnthropicRequestContentBlock {
                content_type: "tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: tool_name,
                input: None,
                tool_use_id: tool_call_id,
                content: Some(tool_result.unwrap_or_else(|| Value::String(content))),
                is_error: tool_error,
            }],
        };
    }

    let mut blocks = Vec::new();

    if let Some(thinking) = thinking {
        if thinking.text.is_some() || thinking.signature.is_some() {
            blocks.push(AnthropicRequestContentBlock {
                content_type: "thinking".to_string(),
                text: None,
                thinking: Some(thinking.text.unwrap_or_default()),
                signature: thinking.signature,
                data: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                is_error: None,
            });
        }

        if let Some(redacted) = thinking.redacted {
            blocks.push(AnthropicRequestContentBlock {
                content_type: "redacted_thinking".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: Some(redacted),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                is_error: None,
            });
        }
    }

    for tool_call in tool_calls {
        blocks.push(AnthropicRequestContentBlock {
            content_type: "tool_use".to_string(),
            text: None,
            thinking: None,
            signature: None,
            data: None,
            id: Some(tool_call.id),
            name: Some(tool_call.name),
            input: Some(tool_call.arguments),
            tool_use_id: None,
            content: None,
            is_error: None,
        });
    }

    if !content.is_empty() {
        blocks.push(AnthropicRequestContentBlock {
            content_type: "text".to_string(),
            text: Some(content),
            thinking: None,
            signature: None,
            data: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            is_error: None,
        });
    }

    AnthropicRequestMessage {
        role,
        content: blocks,
    }
}

pub(super) fn convert_thinking_config(
    thinking: Option<&ThinkingConfig>,
) -> Option<AnthropicThinkingRequest> {
    let thinking = thinking?;
    if !thinking.enabled {
        return None;
    }

    Some(AnthropicThinkingRequest {
        thinking_type: "enabled",
        budget_tokens: thinking.budget_tokens.or(Some(1024)),
        display: match thinking.display {
            Some(ThinkingDisplay::Summarized) => Some("summarized"),
            Some(ThinkingDisplay::Omitted) => Some("omitted"),
            None => None,
        },
    })
}

pub(super) fn convert_request(request: ChatRequest) -> AnthropicRequest {
    let ChatRequest {
        model,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        system,
        thinking,
    } = request;

    AnthropicRequest {
        model,
        messages: messages.into_iter().map(convert_request_message).collect(),
        tools: convert_tools(&tools),
        tool_choice: convert_tool_choice(tool_choice.as_ref()),
        max_tokens: max_tokens.unwrap_or(4096),
        system,
        temperature,
        stream: None,
        thinking: convert_thinking_config(thinking.as_ref()),
    }
}

pub(super) fn convert_response(
    response: AnthropicResponse,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let content = response
        .content
        .iter()
        .filter(|content| content.content_type == "text")
        .filter_map(|content| content.text.clone())
        .collect::<Vec<_>>()
        .join("");

    let mut thinking_output = ThinkingOutput::default();
    let mut tool_calls = Vec::new();
    for content_block in &response.content {
        match content_block.content_type.as_str() {
            "thinking" => {
                thinking_output.text = content_block.thinking.clone();
                thinking_output.signature = content_block.signature.clone();
            }
            "redacted_thinking" => {
                thinking_output.redacted = content_block.data.clone();
            }
            "tool_use" => {
                if let (Some(id), Some(name), Some(input)) = (
                    content_block.id.clone(),
                    content_block.name.clone(),
                    content_block.input.clone(),
                ) {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
            }
            _ => {}
        }
    }

    let thinking = if thinking_output.is_empty() {
        None
    } else {
        Some(thinking_output)
    };

    ChatResponse {
        id: response.id,
        content,
        model: response.model,
        usage: Usage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
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
