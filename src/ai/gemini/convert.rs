use super::protocol::{
    GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiFunctionCallingConfig,
    GeminiFunctionDeclaration, GeminiFunctionResponse, GeminiGenerationConfig, GeminiPart,
    GeminiRequest, GeminiResponse, GeminiThinkingConfig, GeminiTool, GeminiToolConfig,
    GeminiUsageMetadata,
};
use crate::ai::{
    ChatRequest, ChatResponse, DebugTrace, Message, ThinkingOutput, ToolCall, ToolChoice,
    ToolDefinition, Usage,
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

fn convert_message(message: Message) -> GeminiContent {
    let Message {
        role,
        content,
        thinking,
        tool_calls,
        tool_call_id,
        tool_name,
        tool_result,
        tool_error: _,
    } = message;

    let mut parts = Vec::new();

    if matches!(role.as_str(), "assistant" | "model") {
        if let Some(thinking) = &thinking {
            if let Some(text) = &thinking.text {
                parts.push(GeminiPart {
                    text: Some(text.clone()),
                    thought: Some(true),
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                });
            }
        }
    }

    for tool_call in tool_calls {
        parts.push(GeminiPart {
            text: None,
            thought: None,
            thought_signature: None,
            function_call: Some(GeminiFunctionCall {
                name: tool_call.name,
                args: tool_call.arguments,
            }),
            function_response: None,
        });
    }

    if role == "tool" {
        parts.push(GeminiPart {
            text: None,
            thought: None,
            thought_signature: None,
            function_call: None,
            function_response: Some(GeminiFunctionResponse {
                name: tool_name.unwrap_or_else(|| "tool".to_string()),
                response: tool_result_object(
                    tool_result
                        .or_else(|| tool_call_id.map(Value::String))
                        .unwrap_or_else(|| Value::String(content.clone())),
                ),
            }),
        });
    }

    let thought_signature = thinking.and_then(|thinking| thinking.signature);
    if role != "tool" || !content.is_empty() {
        parts.push(GeminiPart {
            text: Some(content),
            thought: None,
            thought_signature,
            function_call: None,
            function_response: None,
        });
    }

    GeminiContent {
        role: Some(convert_role(&role)),
        parts,
    }
}

fn convert_system_instruction(system: String) -> GeminiContent {
    GeminiContent {
        role: Some("system".to_string()),
        parts: vec![GeminiPart {
            text: Some(system),
            thought: None,
            thought_signature: None,
            function_call: None,
            function_response: None,
        }],
    }
}

pub(super) fn convert_request(request: ChatRequest) -> GeminiRequest {
    let ChatRequest {
        model: _,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        system,
        thinking,
    } = request;

    GeminiRequest {
        contents: messages.into_iter().map(convert_message).collect(),
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
    }
}

pub(super) fn parse_candidate(
    candidate: &GeminiCandidate,
) -> (String, ThinkingOutput, Vec<ToolCall>) {
    let mut content = String::new();
    let mut thinking = ThinkingOutput::default();
    let mut tool_calls = Vec::new();

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

    (content, thinking, tool_calls)
}

pub(super) fn convert_response(
    response: GeminiResponse,
    fallback_model: &str,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let candidate = response.candidates.first();
    let (content, thinking, tool_calls) = match candidate {
        Some(candidate) => parse_candidate(candidate),
        None => (String::new(), ThinkingOutput::default(), Vec::new()),
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
