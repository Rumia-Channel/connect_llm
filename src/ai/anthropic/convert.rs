use super::protocol::{
    AnthropicImageSource, AnthropicRequest, AnthropicRequestContentBlock, AnthropicRequestMessage,
    AnthropicResponse, AnthropicThinkingRequest, AnthropicToolChoice, AnthropicToolDefinition,
};
use crate::ai::{
    ChatRequest, ChatResponse, ContentPart, DebugTrace, Message, MultimodalChatRequest,
    RequestMessage, ThinkingConfig, ThinkingDisplay, ThinkingOutput, ToolCall, ToolChoice,
    ToolDefinition, Usage, unsupported_image_url_error,
};

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

fn text_block(text: String) -> AnthropicRequestContentBlock {
    AnthropicRequestContentBlock {
        content_type: "text".to_string(),
        text: Some(text),
        thinking: None,
        signature: None,
        data: None,
        source: None,
        id: None,
        name: None,
        input: None,
        tool_use_id: None,
        content: None,
        is_error: None,
    }
}

fn normalize_anthropic_media_type(media_type: &str) -> Option<&'static str> {
    match media_type.trim().to_ascii_lowercase().as_str() {
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/png" => Some("image/png"),
        "image/gif" => Some("image/gif"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

pub(super) fn convert_request_message(message: Message) -> AnthropicRequestMessage {
    convert_multimodal_request_message(message.into())
        .expect("text-only ChatRequest conversion must not fail")
}

pub(super) fn convert_multimodal_request_message(
    message: RequestMessage,
) -> Result<AnthropicRequestMessage, crate::ai::AiError> {
    match message {
        RequestMessage::Tool {
            tool_call_id,
            tool_name,
            result,
            is_error,
            ..
        } => Ok(AnthropicRequestMessage {
            role: "user".to_string(),
            content: vec![AnthropicRequestContentBlock {
                content_type: "tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                source: None,
                id: None,
                name: Some(tool_name),
                input: None,
                tool_use_id: Some(tool_call_id),
                content: Some(result),
                is_error: is_error.then_some(true),
            }],
        }),
        RequestMessage::User { content, .. } => {
            let mut blocks = Vec::new();
            for part in content {
                match part {
                    ContentPart::Text { text } => blocks.push(text_block(text)),
                    ContentPart::Image { image } => {
                        if let Some((mime_type, data_base64)) = image.as_inline_base64() {
                            let Some(media_type) = normalize_anthropic_media_type(&mime_type)
                            else {
                                return Err(crate::ai::AiError::configuration(
                                    "Anthropic input images must be JPEG, PNG, GIF, or WebP",
                                )
                                .with_provider(crate::ai::AiProvider::Anthropic)
                                .with_operation("chat_multimodal")
                                .with_target("/v1/messages"));
                            };
                            blocks.push(AnthropicRequestContentBlock {
                                content_type: "image".to_string(),
                                text: None,
                                thinking: None,
                                signature: None,
                                data: None,
                                source: Some(AnthropicImageSource {
                                    source_type: "base64",
                                    media_type: Some(media_type.to_string()),
                                    data: Some(data_base64),
                                    url: None,
                                }),
                                id: None,
                                name: None,
                                input: None,
                                tool_use_id: None,
                                content: None,
                                is_error: None,
                            });
                        } else if let Some(url) = image.as_url() {
                            blocks.push(AnthropicRequestContentBlock {
                                content_type: "image".to_string(),
                                text: None,
                                thinking: None,
                                signature: None,
                                data: None,
                                source: Some(AnthropicImageSource {
                                    source_type: "url",
                                    media_type: None,
                                    data: None,
                                    url: Some(url.to_string()),
                                }),
                                id: None,
                                name: None,
                                input: None,
                                tool_use_id: None,
                                content: None,
                                is_error: None,
                            });
                        } else {
                            return Err(unsupported_image_url_error(
                                crate::ai::AiProvider::Anthropic,
                                "chat_multimodal",
                                "/v1/messages",
                            ));
                        }
                    }
                }
            }
            Ok(AnthropicRequestMessage {
                role: "user".to_string(),
                content: blocks,
            })
        }
        RequestMessage::Assistant {
            content,
            thinking,
            tool_calls,
            ..
        } => {
            let mut blocks = Vec::new();

            if let Some(thinking) = thinking {
                if thinking.text.is_some() || thinking.signature.is_some() {
                    blocks.push(AnthropicRequestContentBlock {
                        content_type: "thinking".to_string(),
                        text: None,
                        thinking: Some(thinking.text.unwrap_or_default()),
                        signature: thinking.signature,
                        data: None,
                        source: None,
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
                        source: None,
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
                    source: None,
                    id: Some(tool_call.id),
                    name: Some(tool_call.name),
                    input: Some(tool_call.arguments),
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                });
            }

            if !content.is_empty() {
                blocks.push(text_block(content));
            }

            Ok(AnthropicRequestMessage {
                role: "assistant".to_string(),
                content: blocks,
            })
        }
    }
}

pub(super) fn convert_request(request: ChatRequest) -> AnthropicRequest {
    convert_multimodal_request(request.into())
        .expect("text-only ChatRequest conversion must not fail")
}

pub(super) fn convert_multimodal_request(
    request: MultimodalChatRequest,
) -> Result<AnthropicRequest, crate::ai::AiError> {
    let MultimodalChatRequest {
        model,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        system,
        thinking,
    } = request;

    Ok(AnthropicRequest {
        model,
        messages: messages
            .into_iter()
            .map(convert_multimodal_request_message)
            .collect::<Result<Vec<_>, _>>()?,
        tools: convert_tools(&tools),
        tool_choice: convert_tool_choice(tool_choice.as_ref()),
        max_tokens: max_tokens.unwrap_or(4096),
        system,
        temperature,
        stream: None,
        thinking: convert_thinking_config(thinking.as_ref()),
    })
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
