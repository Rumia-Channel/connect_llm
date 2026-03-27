use super::protocol::{OpenAiCodexEvent, OpenAiCodexResponse, PendingToolCallState};
use crate::ai::{
    AiError, DebugTrace, StreamChunk, ThinkingOutput, ToolCallDelta, capture_debug_text,
};
use std::collections::HashMap;

pub(super) struct ParsedResponseStream {
    pub response: OpenAiCodexResponse,
    pub content: String,
    pub thinking_text: String,
    pub thinking_signature: Option<String>,
    pub pending_tool_calls: HashMap<usize, PendingToolCallState>,
}

fn debug_trace(
    request_debug: &mut Option<String>,
    response_debug: Option<String>,
) -> Option<DebugTrace> {
    if request_debug.is_some() || response_debug.is_some() {
        Some(DebugTrace {
            request: request_debug.take(),
            response: response_debug,
        })
    } else {
        None
    }
}

pub(super) fn parse_response_body(body: &str) -> Result<ParsedResponseStream, AiError> {
    let mut content = String::new();
    let mut thinking_text = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut pending_tool_calls: HashMap<usize, PendingToolCallState> = HashMap::new();
    let mut final_response: Option<OpenAiCodexResponse> = None;

    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.is_empty() || !line.starts_with("data: ") {
            continue;
        }

        let data = &line[6..];
        if data == "[DONE]" {
            break;
        }

        let event: OpenAiCodexEvent =
            serde_json::from_str(data).map_err(|error| AiError::Parse(error.to_string()))?;

        match event.event_type.as_str() {
            "response.output_text.delta" => {
                if let Some(delta) = event.delta {
                    content.push_str(&delta);
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.delta {
                    thinking_text.push_str(&delta);
                }
            }
            "response.output_item.added" => {
                if let (Some(output_index), Some(item)) = (event.output_index, event.item) {
                    if item.item_type == "function_call" {
                        let pending = pending_tool_calls.entry(output_index).or_default();
                        if let Some(call_id) = item.call_id.or(item.id) {
                            pending.id = Some(call_id);
                        }
                        if let Some(name) = item.name {
                            pending.name = Some(name);
                        }
                    } else if item.item_type == "reasoning" && thinking_signature.is_none() {
                        thinking_signature = item.encrypted_content;
                    }
                }
            }
            "response.output_item.done" => {
                if let (Some(output_index), Some(item)) = (event.output_index, event.item) {
                    if item.item_type == "function_call" {
                        let pending = pending_tool_calls.entry(output_index).or_default();
                        if let Some(call_id) = item.call_id.or(item.id) {
                            pending.id = Some(call_id);
                        }
                        if let Some(name) = item.name {
                            pending.name = Some(name);
                        }
                        if !pending.saw_argument_delta {
                            if let Some(arguments) = item.arguments {
                                pending.arguments = arguments;
                            }
                        }
                    } else if item.item_type == "reasoning" && thinking_signature.is_none() {
                        thinking_signature = item.encrypted_content;
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                if let (Some(output_index), Some(delta)) = (event.output_index, event.delta) {
                    let pending = pending_tool_calls.entry(output_index).or_default();
                    pending.saw_argument_delta = true;
                    pending.arguments.push_str(&delta);
                }
            }
            "response.completed" => {
                final_response = event.response;
            }
            _ => {}
        }
    }

    let response = final_response.ok_or_else(|| {
        AiError::Parse("missing response.completed event in Codex stream".to_string())
    })?;

    Ok(ParsedResponseStream {
        response,
        content,
        thinking_text,
        thinking_signature,
        pending_tool_calls,
    })
}

pub(super) fn finalize_response_thinking(
    thinking: &mut Option<ThinkingOutput>,
    thinking_text: String,
    thinking_signature: Option<String>,
) {
    if thinking.is_none() && !thinking_text.is_empty() {
        *thinking = Some(ThinkingOutput {
            text: Some(thinking_text),
            signature: thinking_signature.clone(),
            redacted: None,
        });
    } else if let Some(thinking) = thinking.as_mut() {
        if thinking.signature.is_none() {
            thinking.signature = thinking_signature;
        }
    }
}

pub(super) fn parse_stream_line(
    raw_line: &str,
    request_debug: &mut Option<String>,
    pending_tool_calls: &mut HashMap<usize, PendingToolCallState>,
) -> Result<Option<StreamChunk>, AiError> {
    let line = raw_line.trim();
    if line.is_empty() || !line.starts_with("data: ") {
        return Ok(None);
    }

    let data = &line[6..];
    let response_debug = capture_debug_text("openai_codex stream sse", line.to_string());

    if data == "[DONE]" {
        return Ok(Some(StreamChunk {
            delta: String::new(),
            thinking_delta: None,
            thinking_signature: None,
            images: Vec::new(),
            tool_call_deltas: Vec::new(),
            done: true,
            debug: debug_trace(request_debug, response_debug),
        }));
    }

    let event: OpenAiCodexEvent =
        serde_json::from_str(data).map_err(|error| AiError::Parse(error.to_string()))?;

    Ok(match event.event_type.as_str() {
        "response.output_text.delta" => {
            let delta = event.delta.unwrap_or_default();
            if delta.is_empty() {
                None
            } else {
                Some(StreamChunk {
                    delta,
                    thinking_delta: None,
                    thinking_signature: None,
                    images: Vec::new(),
                    tool_call_deltas: Vec::new(),
                    done: false,
                    debug: debug_trace(request_debug, response_debug),
                })
            }
        }
        "response.output_item.added" => {
            let Some(output_index) = event.output_index else {
                return Ok(None);
            };
            let Some(item) = event.item else {
                return Ok(None);
            };
            if item.item_type != "function_call" {
                return Ok(None);
            }

            let pending = pending_tool_calls.entry(output_index).or_default();
            let id = item.call_id.or(item.id);
            let name = item.name;
            if let Some(id_value) = &id {
                pending.id = Some(id_value.clone());
            }
            if let Some(name_value) = &name {
                pending.name = Some(name_value.clone());
            }

            if id.is_none() && name.is_none() {
                None
            } else {
                Some(StreamChunk {
                    delta: String::new(),
                    thinking_delta: None,
                    thinking_signature: None,
                    images: Vec::new(),
                    tool_call_deltas: vec![ToolCallDelta {
                        index: output_index,
                        id,
                        name,
                        arguments: None,
                    }],
                    done: false,
                    debug: debug_trace(request_debug, response_debug),
                })
            }
        }
        "response.function_call_arguments.delta" => {
            let Some(output_index) = event.output_index else {
                return Ok(None);
            };
            let Some(arguments) = event.delta else {
                return Ok(None);
            };

            let pending = pending_tool_calls.entry(output_index).or_default();
            pending.saw_argument_delta = true;
            pending.arguments.push_str(&arguments);

            Some(StreamChunk {
                delta: String::new(),
                thinking_delta: None,
                thinking_signature: None,
                images: Vec::new(),
                tool_call_deltas: vec![ToolCallDelta {
                    index: output_index,
                    id: None,
                    name: None,
                    arguments: Some(arguments),
                }],
                done: false,
                debug: debug_trace(request_debug, response_debug),
            })
        }
        "response.output_item.done" => {
            let Some(output_index) = event.output_index else {
                return Ok(None);
            };
            let Some(item) = event.item else {
                return Ok(None);
            };

            if item.item_type == "reasoning" {
                if let Some(signature) = item.encrypted_content {
                    Some(StreamChunk {
                        delta: String::new(),
                        thinking_delta: None,
                        thinking_signature: Some(signature),
                        images: Vec::new(),
                        tool_call_deltas: Vec::new(),
                        done: false,
                        debug: debug_trace(request_debug, response_debug),
                    })
                } else {
                    None
                }
            } else if item.item_type != "function_call" {
                None
            } else {
                let pending = pending_tool_calls.entry(output_index).or_default();
                let id = item.call_id.or(item.id);
                let name = item.name;
                let arguments = if pending.saw_argument_delta {
                    None
                } else {
                    item.arguments
                };

                if let Some(id_value) = &id {
                    pending.id = Some(id_value.clone());
                }
                if let Some(name_value) = &name {
                    pending.name = Some(name_value.clone());
                }
                if let Some(arguments_value) = &arguments {
                    pending.arguments = arguments_value.clone();
                }

                if id.is_none() && name.is_none() && arguments.is_none() {
                    None
                } else {
                    Some(StreamChunk {
                        delta: String::new(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: vec![ToolCallDelta {
                            index: output_index,
                            id,
                            name,
                            arguments,
                        }],
                        done: false,
                        debug: debug_trace(request_debug, response_debug),
                    })
                }
            }
        }
        "response.reasoning_summary_text.delta" => {
            let thinking_delta = event.delta.or(event.text);
            thinking_delta.map(|thinking_delta| StreamChunk {
                delta: String::new(),
                thinking_delta: Some(thinking_delta),
                thinking_signature: None,
                images: Vec::new(),
                tool_call_deltas: Vec::new(),
                done: false,
                debug: debug_trace(request_debug, response_debug),
            })
        }
        "response.completed" => Some(StreamChunk {
            delta: String::new(),
            thinking_delta: None,
            thinking_signature: None,
            images: Vec::new(),
            tool_call_deltas: Vec::new(),
            done: true,
            debug: debug_trace(request_debug, response_debug),
        }),
        _ => None,
    })
}
