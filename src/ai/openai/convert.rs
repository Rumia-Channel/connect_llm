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

#[derive(Debug, Default, Clone)]
pub(super) struct EmbeddedThoughtState {
    inside_thought: bool,
    pending: String,
}

impl EmbeddedThoughtState {
    pub(super) fn push(&mut self, chunk: &str) -> (String, Option<String>) {
        self.pending.push_str(chunk);
        let mut content = String::new();
        let mut thinking = String::new();

        loop {
            if self.inside_thought {
                if let Some(end) = self.pending.find("</thought>") {
                    thinking.push_str(&self.pending[..end]);
                    self.pending.drain(..end + "</thought>".len());
                    self.inside_thought = false;
                    continue;
                }

                let flush_len = safe_flush_len(&self.pending, "</thought>");
                if flush_len > 0 {
                    thinking.push_str(&self.pending[..flush_len]);
                    self.pending.drain(..flush_len);
                }
                break;
            }

            if let Some(start) = self.pending.find("<thought>") {
                content.push_str(&self.pending[..start]);
                self.pending.drain(..start + "<thought>".len());
                self.inside_thought = true;
                continue;
            }

            let flush_len = safe_flush_len(&self.pending, "<thought>");
            if flush_len > 0 {
                content.push_str(&self.pending[..flush_len]);
                self.pending.drain(..flush_len);
            }
            break;
        }

        let thinking = (!thinking.is_empty()).then_some(thinking);
        (content, thinking)
    }

    pub(super) fn finish(mut self) -> (String, Option<String>) {
        let tail = std::mem::take(&mut self.pending);
        if tail.is_empty() {
            return (String::new(), None);
        }

        if self.inside_thought {
            (String::new(), Some(tail))
        } else {
            (tail, None)
        }
    }

    pub(super) fn push_google_chunk(
        &mut self,
        chunk: &str,
        google_thought: bool,
    ) -> (String, Option<String>) {
        if google_thought {
            self.inside_thought = true;
            if let Some(end) = chunk.find("</thought>") {
                let thinking = strip_thought_tags(&chunk[..end]);
                let visible = chunk[end + "</thought>".len()..].to_string();
                self.inside_thought = false;
                return (visible, (!thinking.is_empty()).then_some(thinking));
            }

            let thinking = strip_thought_tags(chunk);
            return (String::new(), (!thinking.is_empty()).then_some(thinking));
        }

        if self.inside_thought {
            if let Some(end) = chunk.find("</thought>") {
                let thinking = strip_thought_tags(&chunk[..end]);
                let visible = chunk[end + "</thought>".len()..].to_string();
                self.inside_thought = false;
                return (visible, (!thinking.is_empty()).then_some(thinking));
            }

            let thinking = strip_thought_tags(chunk);
            return (String::new(), (!thinking.is_empty()).then_some(thinking));
        }

        split_embedded_thoughts(chunk)
    }
}

fn safe_flush_len(pending: &str, tag: &str) -> usize {
    let keep_bytes = tag.len().saturating_sub(1);
    let max_flush = pending.len().saturating_sub(keep_bytes);
    floor_char_boundary(pending, max_flush)
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn split_embedded_thoughts(content: &str) -> (String, Option<String>) {
    let mut state = EmbeddedThoughtState::default();
    let (mut visible, mut thinking) = state.push(content);
    let (tail_visible, tail_thinking) = state.finish();
    visible.push_str(&tail_visible);
    match (thinking.as_mut(), tail_thinking) {
        (Some(existing), Some(tail)) => existing.push_str(&tail),
        (None, Some(tail)) => thinking = Some(tail),
        _ => {}
    }
    (visible, thinking)
}

fn strip_thought_tags(text: &str) -> String {
    text.replace("<thought>", "").replace("</thought>", "")
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
            created_at_ms: _,
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
    base_url: &str,
    request_debug: Option<String>,
    response_debug: Option<String>,
) -> ChatResponse {
    let tool_calls = response
        .choices
        .first()
        .map(|choice| convert_tool_calls_to_response(choice.message.tool_calls.clone()))
        .unwrap_or_default();
    let mut thinking = response
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
            signature: response
                .choices
                .first()
                .and_then(|choice| choice.message.extra_content.as_ref())
                .and_then(|extra| extra.google.as_ref())
                .and_then(|google| google.thought_signature.clone()),
            redacted: None,
        });
    let raw_content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .unwrap_or_default();
    let (content, embedded_thinking) = if is_google_openai_compat(base_url) {
        split_embedded_thoughts(&raw_content)
    } else {
        (raw_content, None)
    };

    if let Some(embedded_thinking) = embedded_thinking {
        match thinking.as_mut() {
            Some(existing) => match existing.text.as_mut() {
                Some(text) => text.push_str(&embedded_thinking),
                None => existing.text = Some(embedded_thinking),
            },
            None => {
                thinking = Some(ThinkingOutput {
                    text: Some(embedded_thinking),
                    signature: None,
                    redacted: None,
                })
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::{EmbeddedThoughtState, split_embedded_thoughts};

    #[test]
    fn split_embedded_thoughts_removes_wrapped_content() {
        let (content, thinking) =
            split_embedded_thoughts("<thought>abc</thought>hello<thought>def</thought>");
        assert_eq!(content, "hello");
        assert_eq!(thinking.as_deref(), Some("abcdef"));
    }

    #[test]
    fn embedded_thought_state_handles_split_tags() {
        let mut state = EmbeddedThoughtState::default();
        let (content_1, thinking_1) = state.push("<tho");
        assert!(content_1.is_empty());
        assert!(thinking_1.is_none());

        let (content_2, thinking_2) = state.push("ught>abc</tho");
        assert!(content_2.is_empty());
        assert!(thinking_2.is_none());

        let (content_3, thinking_3) = state.push("ught>hello");
        assert!(content_3.is_empty());
        assert_eq!(thinking_3.as_deref(), Some("abc"));

        let (tail_content, tail_thinking) = state.finish();
        assert_eq!(tail_content, "hello");
        assert!(tail_thinking.is_none());
    }

    #[test]
    fn split_embedded_thoughts_preserves_utf8_boundaries() {
        let mut state = EmbeddedThoughtState::default();
        let (content_1, thinking_1) = state.push("<thought>こんにちは");
        assert!(content_1.is_empty());
        assert_eq!(thinking_1.as_deref(), Some("こん"));

        let (content_2, thinking_2) = state.push("</thought>世界");
        assert!(content_2.is_empty());
        assert_eq!(thinking_2.as_deref(), Some("にちは"));

        let (tail_content, tail_thinking) = state.finish();
        assert_eq!(tail_content, "世界");
        assert!(tail_thinking.is_none());
    }

    #[test]
    fn google_thought_chunks_do_not_leak_into_visible_text() {
        let mut state = EmbeddedThoughtState::default();
        let (content_1, thinking_1) = state.push_google_chunk("<thought>hello phr", true);
        assert!(content_1.is_empty());
        assert_eq!(thinking_1.as_deref(), Some("hello phr"));

        let (content_2, thinking_2) = state.push_google_chunk("asing", true);
        assert!(content_2.is_empty());
        assert_eq!(thinking_2.as_deref(), Some("asing"));

        let (content_3, thinking_3) = state.push_google_chunk("</thought>visible text", false);
        assert_eq!(content_3, "visible text");
        assert!(thinking_3.is_none());
    }
}
