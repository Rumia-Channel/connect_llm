use connect_llm::{ChatRequest, ChatResponse, DebugTrace, ThinkingOutput, ToolCall, Usage};
use futures_util::StreamExt;
use std::io::{self, Write};

pub(crate) async fn send_request(
    client: &dyn connect_llm::AiClient,
    request: ChatRequest,
    use_stream: bool,
    include_thinking: bool,
) -> Result<ChatResponse, connect_llm::AiError> {
    if !use_stream {
        let mut response = client.chat(request).await?;
        if !include_thinking {
            response.thinking = None;
        }
        return Ok(response);
    }

    let model = request.model.clone();
    let mut stream = client.chat_stream(request);
    let mut content = String::new();
    let mut thinking_text = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut tool_calls: Vec<(Option<String>, Option<String>, String)> = Vec::new();
    let mut printed_assistant_prefix = false;
    let mut printed_thinking_prefix = false;
    let mut debug_request: Option<String> = None;
    let mut debug_responses: Vec<String> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        if let Some(debug) = chunk.debug {
            if debug_request.is_none() {
                debug_request = debug.request;
            }
            if let Some(response) = debug.response {
                debug_responses.push(response);
            }
        }

        for tool_call_delta in chunk.tool_call_deltas {
            while tool_calls.len() <= tool_call_delta.index {
                tool_calls.push((None, None, String::new()));
            }

            if let Some(entry) = tool_calls.get_mut(tool_call_delta.index) {
                if let Some(id) = tool_call_delta.id {
                    entry.0 = Some(id);
                }
                if let Some(name) = tool_call_delta.name {
                    entry.1 = Some(name);
                }
                if let Some(arguments) = tool_call_delta.arguments {
                    entry.2.push_str(&arguments);
                }
            }
        }

        if !chunk.delta.is_empty() {
            if !printed_assistant_prefix {
                print!("\nassistant> ");
                printed_assistant_prefix = true;
            }
            print!("{}", chunk.delta);
            io::stdout()
                .flush()
                .map_err(|error| connect_llm::AiError::Http(error.to_string()))?;
            content.push_str(&chunk.delta);
        }

        if include_thinking {
            if let Some(thinking_delta) = chunk.thinking_delta {
                if !printed_thinking_prefix {
                    print!("\nthinking> ");
                    printed_thinking_prefix = true;
                }
                print!("{}", thinking_delta);
                io::stdout()
                    .flush()
                    .map_err(|error| connect_llm::AiError::Http(error.to_string()))?;
                thinking_text.push_str(&thinking_delta);
            }
            if let Some(signature) = chunk.thinking_signature {
                thinking_signature = Some(signature);
            }
        }

        if chunk.done {
            break;
        }
    }

    if printed_assistant_prefix || printed_thinking_prefix {
        println!();
    }

    Ok(ChatResponse {
        id: "stream".to_string(),
        content,
        model,
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        thinking: if include_thinking && !thinking_text.is_empty() {
            Some(ThinkingOutput {
                text: Some(thinking_text),
                signature: thinking_signature,
                redacted: None,
            })
        } else if include_thinking && thinking_signature.is_some() {
            Some(ThinkingOutput {
                text: None,
                signature: thinking_signature,
                redacted: None,
            })
        } else {
            None
        },
        tool_calls: tool_calls
            .into_iter()
            .enumerate()
            .filter_map(|(index, (id, name, arguments))| {
                name.map(|name| ToolCall {
                    id: id.unwrap_or_else(|| format!("tool-call-{}", index)),
                    name,
                    arguments: serde_json::from_str(&arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(arguments)),
                })
            })
            .collect(),
        debug: if debug_request.is_some() || !debug_responses.is_empty() {
            Some(DebugTrace {
                request: debug_request,
                response: if debug_responses.is_empty() {
                    None
                } else {
                    Some(debug_responses.join("\n"))
                },
            })
        } else {
            None
        },
    })
}
