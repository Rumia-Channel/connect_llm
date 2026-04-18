use super::protocol::{
    GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiFunctionCallingConfig,
    GeminiFunctionDeclaration, GeminiFunctionResponse, GeminiGenerationConfig, GeminiPart,
    GeminiRequest, GeminiResponse, GeminiThinkingConfig, GeminiTool, GeminiToolConfig,
    GeminiUsageMetadata,
};
use crate::ai::{
    ChatRequest, ChatResponse, ContentPart, DebugTrace, GeneratedImage, MultimodalChatRequest,
    RequestMessage, ThinkingOutput, ToolCall, ToolChoice, ToolDefinition, Usage,
    unsupported_image_url_error,
};
use serde_json::{Map, Value};

fn convert_role(role: &str) -> String {
    match role {
        "tool" => "tool".to_string(),
        "assistant" | "model" => "model".to_string(),
        _ => "user".to_string(),
    }
}

fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<GeminiTool>> {
    if tools.is_empty() {
        return None;
    }

    Some(vec![GeminiTool {
        function_declarations: tools
            .iter()
            .map(|tool| GeminiFunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            })
            .collect(),
    }])
}

fn convert_tool_config(choice: Option<&ToolChoice>) -> Option<GeminiToolConfig> {
    match choice? {
        ToolChoice::Auto => Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: "AUTO",
                allowed_function_names: None,
            },
        }),
        ToolChoice::None => Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: "NONE",
                allowed_function_names: None,
            },
        }),
        ToolChoice::Required => Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: None,
            },
        }),
        ToolChoice::Tool(name) => Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: Some(vec![name.clone()]),
            },
        }),
    }
}

fn tool_result_object(value: Value) -> Value {
    match value {
        Value::Object(_) => value,
        other => {
            let mut object = Map::new();
            object.insert("value".to_string(), other);
            Value::Object(object)
        }
    }
}

fn text_part(text: String) -> GeminiPart {
    GeminiPart {
        text: Some(text),
        inline_data: None,
        thought: None,
        thought_signature: None,
        function_call: None,
        function_response: None,
    }
}

fn convert_message(message: crate::ai::Message) -> GeminiContent {
    convert_multimodal_message(message.into())
        .expect("text-only ChatRequest conversion must not fail")
}

fn convert_multimodal_message(
    message: RequestMessage,
) -> Result<GeminiContent, crate::ai::AiError> {
    match message {
        RequestMessage::Tool {
            tool_call_id,
            tool_name,
            result,
            ..
        } => Ok(GeminiContent {
            role: Some(convert_role("tool")),
            parts: vec![GeminiPart {
                text: None,
                inline_data: None,
                thought: None,
                thought_signature: None,
                function_call: None,
                function_response: Some(GeminiFunctionResponse {
                    name: tool_name,
                    response: tool_result_object({
                        let mut response = match tool_result_object(result) {
                            Value::Object(map) => map,
                            _ => unreachable!(),
                        };
                        response.insert("tool_call_id".to_string(), Value::String(tool_call_id));
                        Value::Object(response)
                    }),
                }),
            }],
        }),
        RequestMessage::User { content, .. } => {
            let mut parts = Vec::new();
            for part in content {
                match part {
                    ContentPart::Text { text } => parts.push(text_part(text)),
                    ContentPart::Image { image } => {
                        if let Some((mime_type, data_base64)) = image.as_inline_base64() {
                            parts.push(GeminiPart {
                                text: None,
                                inline_data: Some(super::protocol::GeminiInlineData {
                                    mime_type: Some(mime_type),
                                    data: data_base64,
                                }),
                                thought: None,
                                thought_signature: None,
                                function_call: None,
                                function_response: None,
                            });
                        } else {
                            return Err(unsupported_image_url_error(
                                crate::ai::AiProvider::Gemini,
                                "chat_multimodal",
                                "generateContent",
                            ));
                        }
                    }
                }
            }
            Ok(GeminiContent {
                role: Some(convert_role("user")),
                parts,
            })
        }
        RequestMessage::Assistant {
            content,
            thinking,
            tool_calls,
            ..
        } => {
            let mut parts = Vec::new();

            if let Some(thinking) = &thinking {
                if let Some(text) = &thinking.text {
                    parts.push(GeminiPart {
                        text: Some(text.clone()),
                        inline_data: None,
                        thought: Some(true),
                        thought_signature: None,
                        function_call: None,
                        function_response: None,
                    });
                }
            }

            for tool_call in tool_calls {
                parts.push(GeminiPart {
                    text: None,
                    inline_data: None,
                    thought: None,
                    thought_signature: None,
                    function_call: Some(GeminiFunctionCall {
                        name: tool_call.name,
                        args: tool_call.arguments,
                    }),
                    function_response: None,
                });
            }

            let thought_signature = thinking.and_then(|thinking| thinking.signature);
            if !content.is_empty() || parts.is_empty() {
                parts.push(GeminiPart {
                    thought_signature,
                    ..text_part(content)
                });
            }

            Ok(GeminiContent {
                role: Some(convert_role("assistant")),
                parts,
            })
        }
    }
}

fn convert_system_instruction(system: String) -> GeminiContent {
    GeminiContent {
        role: Some("system".to_string()),
        parts: vec![GeminiPart {
            text: Some(system),
            inline_data: None,
            thought: None,
            thought_signature: None,
            function_call: None,
            function_response: None,
        }],
    }
}

pub(super) fn convert_request(request: ChatRequest) -> GeminiRequest {
    convert_multimodal_request(request.into())
        .expect("text-only ChatRequest conversion must not fail")
}

pub(super) fn convert_multimodal_request(
    request: MultimodalChatRequest,
) -> Result<GeminiRequest, crate::ai::AiError> {
    let MultimodalChatRequest {
        model: _,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        system,
        thinking,
    } = request;

    Ok(GeminiRequest {
        contents: messages
            .into_iter()
            .map(convert_multimodal_message)
            .collect::<Result<Vec<_>, _>>()?,
        system_instruction: system.map(convert_system_instruction),
        tools: convert_tools(&tools),
        tool_config: convert_tool_config(tool_choice.as_ref()),
        generation_config: Some(GeminiGenerationConfig {
            max_output_tokens: max_tokens,
            temperature,
            thinking_config: thinking.and_then(|thinking| {
                if !thinking.enabled {
                    return None;
                }

                Some(GeminiThinkingConfig {
                    include_thoughts: true,
                    thinking_budget: thinking.budget_tokens,
                })
            }),
        }),
    })
}

pub(super) fn parse_candidate(
    candidate: &GeminiCandidate,
) -> (String, ThinkingOutput, Vec<ToolCall>, Vec<GeneratedImage>) {
    let mut content = String::new();
    let mut thinking = ThinkingOutput::default();
    let mut tool_calls = Vec::new();
    let mut images = Vec::new();

    if let Some(candidate_content) = &candidate.content {
        for part in &candidate_content.parts {
            if let Some(signature) = &part.thought_signature {
                thinking.signature = Some(signature.clone());
            }

            if let Some(function_call) = &part.function_call {
                tool_calls.push(ToolCall {
                    id: format!("gemini-call-{}", tool_calls.len()),
                    name: function_call.name.clone(),
                    arguments: function_call.args.clone(),
                });
                continue;
            }

            if let Some(inline_data) = &part.inline_data {
                images.push(GeneratedImage {
                    mime_type: inline_data.mime_type.clone(),
                    data_base64: Some(inline_data.data.clone()),
                    url: None,
                    revised_prompt: None,
                });
                continue;
            }

            let Some(text) = &part.text else {
                continue;
            };

            if part.thought.unwrap_or(false) {
                match &mut thinking.text {
                    Some(existing) => existing.push_str(text),
                    None => thinking.text = Some(text.clone()),
                }
            } else {
                content.push_str(text);
            }
        }
    }

    (content, thinking, tool_calls, images)
}

pub(super) fn convert_response(
    response: GeminiResponse,
    fallback_model: &str,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let candidate = response.candidates.first();
    let (content, thinking, tool_calls, images) = match candidate {
        Some(candidate) => parse_candidate(candidate),
        None => (
            String::new(),
            ThinkingOutput::default(),
            Vec::new(),
            Vec::new(),
        ),
    };

    let thinking = if thinking.is_empty() {
        None
    } else {
        Some(thinking)
    };

    let usage = response.usage_metadata.unwrap_or(GeminiUsageMetadata {
        prompt_token_count: 0,
        candidates_token_count: 0,
        thoughts_token_count: None,
    });

    ChatResponse {
        id: response.response_id.unwrap_or_default(),
        content,
        model: response
            .model_version
            .unwrap_or_else(|| fallback_model.to_string()),
        usage: Usage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count + usage.thoughts_token_count.unwrap_or(0),
        },
        thinking,
        images,
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

#[cfg(test)]
mod tests {
    use super::{convert_multimodal_request, parse_candidate};
    use crate::ai::gemini::protocol::{
        GeminiCandidate, GeminiContent, GeminiInlineData, GeminiPart,
    };
    use crate::ai::{
        AiErrorKind, AiProvider, ContentPart, InputImage, MultimodalChatRequest, RequestMessage,
    };

    #[test]
    fn parse_candidate_extracts_inline_images() {
        let candidate = GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: Some("image/png".to_string()),
                        data: "aGVsbG8=".to_string(),
                    }),
                    thought: None,
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                }],
            }),
            finish_reason: Some("STOP".to_string()),
        };

        let (content, thinking, tool_calls, images) = parse_candidate(&candidate);
        assert!(content.is_empty());
        assert!(thinking.is_empty());
        assert!(tool_calls.is_empty());
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type.as_deref(), Some("image/png"));
        assert_eq!(images[0].data_base64.as_deref(), Some("aGVsbG8="));
    }

    #[test]
    fn convert_multimodal_request_encodes_inline_images() {
        let request = MultimodalChatRequest::new(
            "gemini-2.5-flash",
            vec![RequestMessage::user_parts(vec![
                ContentPart::text("describe"),
                ContentPart::image(InputImage::from_base64("image/png", "aGVsbG8=")),
            ])],
        );

        let converted = convert_multimodal_request(request).expect("convert request");
        assert_eq!(converted.contents[0].parts.len(), 2);
        assert_eq!(
            converted.contents[0].parts[1]
                .inline_data
                .as_ref()
                .and_then(|data| data.mime_type.as_deref()),
            Some("image/png")
        );
    }

    #[test]
    fn convert_multimodal_request_rejects_remote_image_urls() {
        let request = MultimodalChatRequest::new(
            "gemini-2.5-flash",
            vec![RequestMessage::user_parts(vec![ContentPart::image(
                InputImage::from_url("https://example.com/cat.png"),
            )])],
        );

        let error = convert_multimodal_request(request).expect_err("url images should fail");
        assert_eq!(error.kind, AiErrorKind::Configuration);
        assert_eq!(error.provider, Some(AiProvider::Gemini));
    }
}
