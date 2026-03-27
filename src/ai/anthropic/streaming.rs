use super::protocol::AnthropicStreamResponse;
use crate::ai::{AiError, DebugTrace, StreamChunk, ToolCallDelta, capture_debug_text};

#[derive(Default)]
pub(super) struct AnthropicSseState {
    pending_event: Option<String>,
    pending_data: Option<String>,
    pending_debug_lines: Vec<String>,
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

pub(super) fn consume_line(
    line: &str,
    state: &mut AnthropicSseState,
    request_debug: &mut Option<String>,
) -> Result<Option<StreamChunk>, AiError> {
    let line_trimmed = line.trim_end_matches('\r').trim_end();
    if !line_trimmed.is_empty() {
        let _ = capture_debug_text("anthropic stream sse", line_trimmed.to_string());
        state.pending_debug_lines.push(line_trimmed.to_string());
    }

    if line_trimmed.is_empty() {
        let response_debug = if state.pending_debug_lines.is_empty() {
            None
        } else {
            Some(state.pending_debug_lines.join("\n"))
        };
        state.pending_debug_lines.clear();

        let (Some(event_type), Some(data)) =
            (state.pending_event.take(), state.pending_data.take())
        else {
            return Ok(None);
        };

        return match event_type.as_str() {
            "message_stop" => Ok(Some(StreamChunk {
                delta: String::new(),
                thinking_delta: None,
                thinking_signature: None,
                images: Vec::new(),
                tool_call_deltas: Vec::new(),
                done: true,
                debug: debug_trace(request_debug, response_debug),
            })),
            "error" => Err(AiError::Api(data)),
            "content_block_delta" => {
                convert_content_block_delta(data, response_debug, request_debug)
            }
            "content_block_start" => {
                convert_content_block_start(data, response_debug, request_debug)
            }
            _ => Ok(None),
        };
    }

    if line_trimmed.starts_with("event:") {
        state.pending_event = Some(line_trimmed[6..].trim_start().to_string());
    } else if line_trimmed.starts_with("data:") {
        state.pending_data = Some(line_trimmed[5..].trim_start().to_string());
    }

    Ok(None)
}

fn convert_content_block_delta(
    data: String,
    response_debug: Option<String>,
    request_debug: &mut Option<String>,
) -> Result<Option<StreamChunk>, AiError> {
    let stream_response: AnthropicStreamResponse =
        serde_json::from_str(&data).map_err(|error| AiError::Parse(error.to_string()))?;
    let Some(delta) = stream_response.delta else {
        return Ok(None);
    };

    Ok(match delta.delta_type.as_deref() {
        Some("text_delta") => delta.text.map(|text| StreamChunk {
            delta: text,
            thinking_delta: None,
            thinking_signature: None,
            images: Vec::new(),
            tool_call_deltas: Vec::new(),
            done: false,
            debug: debug_trace(request_debug, response_debug),
        }),
        Some("thinking_delta") => delta.thinking.map(|thinking| StreamChunk {
            delta: String::new(),
            thinking_delta: Some(thinking),
            thinking_signature: None,
            images: Vec::new(),
            tool_call_deltas: Vec::new(),
            done: false,
            debug: debug_trace(request_debug, response_debug),
        }),
        Some("signature_delta") => delta.signature.map(|signature| StreamChunk {
            delta: String::new(),
            thinking_delta: None,
            thinking_signature: Some(signature),
            images: Vec::new(),
            tool_call_deltas: Vec::new(),
            done: false,
            debug: debug_trace(request_debug, response_debug),
        }),
        Some("input_json_delta") => delta.partial_json.map(|arguments| StreamChunk {
            delta: String::new(),
            thinking_delta: None,
            thinking_signature: None,
            images: Vec::new(),
            tool_call_deltas: vec![ToolCallDelta {
                index: stream_response.index.unwrap_or(0),
                id: None,
                name: None,
                arguments: Some(arguments),
            }],
            done: false,
            debug: debug_trace(request_debug, response_debug),
        }),
        _ => None,
    })
}

fn convert_content_block_start(
    data: String,
    response_debug: Option<String>,
    request_debug: &mut Option<String>,
) -> Result<Option<StreamChunk>, AiError> {
    let stream_response: AnthropicStreamResponse =
        serde_json::from_str(&data).map_err(|error| AiError::Parse(error.to_string()))?;
    let Some(content_block) = stream_response.content_block else {
        return Ok(None);
    };
    if content_block.content_type != "tool_use" {
        return Ok(None);
    }

    Ok(Some(StreamChunk {
        delta: String::new(),
        thinking_delta: None,
        thinking_signature: None,
        images: Vec::new(),
        tool_call_deltas: vec![ToolCallDelta {
            index: stream_response.index.unwrap_or(0),
            id: content_block.id,
            name: content_block.name,
            arguments: content_block.input.map(|input| input.to_string()),
        }],
        done: false,
        debug: debug_trace(request_debug, response_debug),
    }))
}
