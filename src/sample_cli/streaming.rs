use connect_llm::{
    ChatRequest, ChatResponse, DebugTrace, GeneratedImage, McpManagedChatResponse, McpRuntime,
    McpStreamEvent, MultimodalChatRequest, ThinkingOutput, ToolCall, Usage,
};
use futures_util::StreamExt;
use std::{
    collections::HashSet,
    io::{self, Write},
};

use super::io::print_mcp_tool_execution;

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
    let mut state = PrintedStreamState::default();
    let mut response_builder = StreamResponseBuilder::new(model.clone());

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print_stream_chunk(&chunk, include_thinking, &mut state)?;
        response_builder.ingest(&chunk);
    }

    finish_stream_output(&state);
    Ok(response_builder.finish(include_thinking))
}

pub(crate) async fn send_multimodal_request(
    client: &dyn connect_llm::AiClient,
    request: MultimodalChatRequest,
    use_stream: bool,
    include_thinking: bool,
) -> Result<ChatResponse, connect_llm::AiError> {
    if !use_stream {
        let mut response = client.chat_multimodal(request).await?;
        if !include_thinking {
            response.thinking = None;
        }
        return Ok(response);
    }

    let model = request.model.clone();
    let mut stream = client.chat_multimodal_stream(request);
    let mut state = PrintedStreamState::default();
    let mut response_builder = StreamResponseBuilder::new(model);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print_stream_chunk(&chunk, include_thinking, &mut state)?;
        response_builder.ingest(&chunk);
    }

    finish_stream_output(&state);
    Ok(response_builder.finish(include_thinking))
}

pub(crate) async fn send_mcp_request(
    runtime: &mut McpRuntime,
    context_manager: &connect_llm::ContextManager,
    client: &dyn connect_llm::AiClient,
    request: ChatRequest,
    include_thinking: bool,
    debug_enabled: bool,
) -> Result<McpManagedChatResponse, connect_llm::AiError> {
    let mut stream = runtime.chat_stream_with_context_manager(context_manager, client, request);
    let mut state = PrintedStreamState::default();
    let mut finished = None;

    while let Some(event) = stream.next().await {
        match event? {
            McpStreamEvent::Chunk(chunk) => {
                print_stream_chunk(&chunk, include_thinking, &mut state)?;
            }
            McpStreamEvent::Finished(managed) => {
                for execution in &managed.tool_executions {
                    print_mcp_tool_execution_event(execution, debug_enabled, &mut state)?;
                }
                print_mcp_finished_fallback(&managed, include_thinking, &mut state)?;
                finished = Some(managed);
                break;
            }
        }
    }

    finish_stream_output(&state);
    finished.ok_or_else(|| {
        connect_llm::AiError::api("MCP stream terminated without a final response".to_string())
    })
}

#[derive(Default)]
struct PrintedStreamState {
    current_section: Option<StreamSection>,
    printed_anything: bool,
    printed_assistant_output: bool,
    printed_thinking_output: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamSection {
    Assistant,
    Thinking,
}

fn print_stream_chunk(
    chunk: &connect_llm::StreamChunk,
    include_thinking: bool,
    state: &mut PrintedStreamState,
) -> Result<(), connect_llm::AiError> {
    if include_thinking {
        if let Some(thinking_delta) = &chunk.thinking_delta {
            begin_stream_section(state, StreamSection::Thinking)?;
            print!("{}", thinking_delta);
            io::stdout()
                .flush()
                .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
            state.printed_thinking_output = true;
        }
    }

    if !chunk.delta.is_empty() {
        begin_stream_section(state, StreamSection::Assistant)?;
        print!("{}", chunk.delta);
        io::stdout()
            .flush()
            .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
        state.printed_assistant_output = true;
    }

    Ok(())
}

fn finish_stream_output(state: &PrintedStreamState) {
    if state.printed_anything {
        println!();
    }
}

fn print_mcp_finished_fallback(
    managed: &McpManagedChatResponse,
    include_thinking: bool,
    state: &mut PrintedStreamState,
) -> Result<(), connect_llm::AiError> {
    if let Some(text) = mcp_fallback_thinking_text(managed, include_thinking, state) {
        begin_stream_section(state, StreamSection::Thinking)?;
        print!("{}", text);
        io::stdout()
            .flush()
            .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
        state.printed_thinking_output = true;
    }

    if !state.printed_assistant_output && !managed.response.content.is_empty() {
        begin_stream_section(state, StreamSection::Assistant)?;
        print!("{}", managed.response.content);
        io::stdout()
            .flush()
            .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
        state.printed_assistant_output = true;
    }

    Ok(())
}

fn mcp_fallback_thinking_text<'a>(
    managed: &'a McpManagedChatResponse,
    include_thinking: bool,
    state: &PrintedStreamState,
) -> Option<&'a str> {
    if !include_thinking || state.printed_thinking_output {
        return None;
    }

    managed
        .response
        .thinking
        .as_ref()
        .and_then(|thinking| thinking.text.as_deref())
        .filter(|text| !text.is_empty())
}

fn print_mcp_tool_execution_event(
    execution: &connect_llm::McpToolExecution,
    debug_enabled: bool,
    state: &mut PrintedStreamState,
) -> Result<(), connect_llm::AiError> {
    finish_active_stream_line(state)?;
    print_mcp_tool_execution(execution, debug_enabled);
    io::stdout()
        .flush()
        .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
    state.current_section = None;
    state.printed_anything = true;
    Ok(())
}

fn begin_stream_section(
    state: &mut PrintedStreamState,
    section: StreamSection,
) -> Result<(), connect_llm::AiError> {
    if state.current_section == Some(section) {
        return Ok(());
    }

    finish_active_stream_line(state)?;
    match section {
        StreamSection::Assistant => print!("assistant> "),
        StreamSection::Thinking => print!("thinking> "),
    }
    io::stdout()
        .flush()
        .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
    state.current_section = Some(section);
    state.printed_anything = true;
    Ok(())
}

fn finish_active_stream_line(state: &mut PrintedStreamState) -> Result<(), connect_llm::AiError> {
    if state.current_section.is_some() {
        println!();
        io::stdout()
            .flush()
            .map_err(|error| connect_llm::AiError::http(error.to_string()))?;
        state.current_section = None;
    }
    Ok(())
}

struct StreamResponseBuilder {
    model: String,
    content: String,
    thinking_text: String,
    thinking_signature: Option<String>,
    tool_calls: Vec<(Option<String>, Option<String>, String)>,
    images: Vec<GeneratedImage>,
    seen_image_keys: HashSet<String>,
    debug_request: Option<String>,
    debug_responses: Vec<String>,
}

impl StreamResponseBuilder {
    fn new(model: String) -> Self {
        Self {
            model,
            content: String::new(),
            thinking_text: String::new(),
            thinking_signature: None,
            tool_calls: Vec::new(),
            images: Vec::new(),
            seen_image_keys: HashSet::new(),
            debug_request: None,
            debug_responses: Vec::new(),
        }
    }

    fn ingest(&mut self, chunk: &connect_llm::StreamChunk) {
        if let Some(debug) = &chunk.debug {
            if self.debug_request.is_none() {
                self.debug_request = debug.request.clone();
            }
            if let Some(response) = &debug.response {
                self.debug_responses.push(response.clone());
            }
        }

        for tool_call_delta in &chunk.tool_call_deltas {
            while self.tool_calls.len() <= tool_call_delta.index {
                self.tool_calls.push((None, None, String::new()));
            }

            if let Some(entry) = self.tool_calls.get_mut(tool_call_delta.index) {
                if let Some(id) = &tool_call_delta.id {
                    entry.0 = Some(id.clone());
                }
                if let Some(name) = &tool_call_delta.name {
                    entry.1 = Some(name.clone());
                }
                if let Some(arguments) = &tool_call_delta.arguments {
                    entry.2.push_str(arguments);
                }
            }
        }

        for image in &chunk.images {
            let key = image.dedup_key();
            if self.seen_image_keys.insert(key) {
                self.images.push(image.clone());
            }
        }

        self.content.push_str(&chunk.delta);
        if let Some(thinking_delta) = &chunk.thinking_delta {
            self.thinking_text.push_str(thinking_delta);
        }
        if let Some(signature) = &chunk.thinking_signature {
            self.thinking_signature = Some(signature.clone());
        }
    }

    fn finish(self, include_thinking: bool) -> ChatResponse {
        ChatResponse {
            id: "stream".to_string(),
            content: self.content,
            model: self.model,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            thinking: if include_thinking && !self.thinking_text.is_empty() {
                Some(ThinkingOutput {
                    text: Some(self.thinking_text),
                    signature: self.thinking_signature,
                    redacted: None,
                })
            } else if include_thinking && self.thinking_signature.is_some() {
                Some(ThinkingOutput {
                    text: None,
                    signature: self.thinking_signature,
                    redacted: None,
                })
            } else {
                None
            },
            images: self.images,
            tool_calls: self
                .tool_calls
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
            debug: if self.debug_request.is_some() || !self.debug_responses.is_empty() {
                Some(DebugTrace {
                    request: self.debug_request,
                    response: if self.debug_responses.is_empty() {
                        None
                    } else {
                        Some(self.debug_responses.join("\n"))
                    },
                })
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PrintedStreamState, mcp_fallback_thinking_text};
    use connect_llm::{ChatResponse, McpManagedChatResponse, ThinkingOutput, Usage};

    #[test]
    fn mcp_fallback_thinking_text_allows_assistant_content() {
        let managed = managed_response("final answer", Some("hidden reasoning"));

        let text = mcp_fallback_thinking_text(&managed, true, &PrintedStreamState::default());

        assert_eq!(text, Some("hidden reasoning"));
    }

    #[test]
    fn mcp_fallback_thinking_text_skips_when_thinking_already_streamed() {
        let managed = managed_response("final answer", Some("hidden reasoning"));
        let state = PrintedStreamState {
            printed_thinking_output: true,
            ..PrintedStreamState::default()
        };

        let text = mcp_fallback_thinking_text(&managed, true, &state);

        assert_eq!(text, None);
    }

    fn managed_response(content: &str, thinking_text: Option<&str>) -> McpManagedChatResponse {
        McpManagedChatResponse {
            response: ChatResponse {
                id: "id".to_string(),
                content: content.to_string(),
                model: "model".to_string(),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
                thinking: thinking_text.map(|text| ThinkingOutput {
                    text: Some(text.to_string()),
                    ..ThinkingOutput::default()
                }),
                images: Vec::new(),
                tool_calls: Vec::new(),
                debug: None,
            },
            compaction: None,
            messages: Vec::new(),
            tool_executions: Vec::new(),
        }
    }
}
