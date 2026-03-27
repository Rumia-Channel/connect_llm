#![allow(dead_code)]

mod auth;
mod protocol;

use self::protocol::{
    OpenAiCodexEvent, OpenAiCodexFunctionCallItem, OpenAiCodexFunctionCallOutputItem,
    OpenAiCodexInputContent, OpenAiCodexInputItem, OpenAiCodexInputMessage, OpenAiCodexOutputItem,
    OpenAiCodexReasoningRequest, OpenAiCodexRequest, OpenAiCodexResponse, OpenAiCodexTool,
    OpenAiCodexToolChoice, PendingToolCallState, api_error_from_response,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingConfig, ThinkingEffort, ThinkingOutput, ToolCall, ToolCallDelta, ToolChoice, Usage,
    capture_debug_json, capture_debug_text, message_tool_result_value, parse_tool_arguments,
    serialize_tool_arguments,
};
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::HashMap;

pub use self::auth::{
    OpenAiCodexBrowserAuth, OpenAiCodexBrowserAuthOptions, login_openai_codex_via_browser,
    openai_codex_auth_path,
};

const DEFAULT_CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
pub struct OpenAiCodexClient {
    client: Client,
    config: AiConfig,
}

impl OpenAiCodexClient {
    pub fn new(config: AiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn endpoint_url(base_url: &str) -> String {
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

    fn convert_tools(tools: Vec<super::ToolDefinition>) -> Option<Vec<OpenAiCodexTool>> {
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

    fn convert_request(request: ChatRequest, stream: bool) -> OpenAiCodexRequest {
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

        let mut input = Vec::new();

        for message in request_messages {
            if message.role == "tool" {
                if let Some(call_id) = message.tool_call_id.clone() {
                    let output = serialize_tool_arguments(&message_tool_result_value(&message));

                    input.push(OpenAiCodexInputItem::FunctionCallOutput(
                        OpenAiCodexFunctionCallOutputItem {
                            item_type: "function_call_output",
                            call_id,
                            output,
                        },
                    ));
                }
                continue;
            }

            if !message.content.is_empty() {
                let content_type = if message.role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                input.push(OpenAiCodexInputItem::Message(OpenAiCodexInputMessage {
                    role: message.role.clone(),
                    content: vec![OpenAiCodexInputContent {
                        content_type,
                        text: message.content.clone(),
                    }],
                }));
            }

            for tool_call in message.tool_calls {
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

        OpenAiCodexRequest {
            model,
            instructions: system.unwrap_or_else(Self::default_instructions),
            input,
            max_output_tokens: None,
            temperature,
            reasoning: Self::convert_reasoning_config(thinking.as_ref()),
            tools: Self::convert_tools(tools),
            tool_choice: Self::convert_tool_choice(tool_choice),
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

    fn convert_response(
        response: OpenAiCodexResponse,
        request_debug: Option<String>,
        response_debug: Option<String>,
    ) -> ChatResponse {
        let usage = response.usage.unwrap_or_default();

        ChatResponse {
            id: response.id,
            content: Self::extract_text_from_output(&response.output),
            model: response.model,
            usage: Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
            },
            thinking: Self::extract_thinking_from_output(&response.output),
            tool_calls: Self::extract_tool_calls_from_output(&response.output),
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
}

#[async_trait::async_trait]
impl AiClient for OpenAiCodexClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = auth::resolve_auth(&self.client, &self.config).await?;
        let url = Self::endpoint_url(&self.config.base_url);
        let request = Self::convert_request(request, true);
        let request_debug =
            capture_debug_json(&format!("openai_codex request POST {}", url), &request);

        let mut builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request);

        if let Some(account_id) = auth.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }

        let response = builder
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;
        let response_debug = capture_debug_text(
            &format!("openai_codex response {} {}", status, url),
            body.clone(),
        );

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

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
        let mut response = Self::convert_response(response, request_debug, response_debug);
        if response.content.is_empty() && !content.is_empty() {
            response.content = content;
        }
        if response.thinking.is_none() && !thinking_text.is_empty() {
            response.thinking = Some(ThinkingOutput {
                text: Some(thinking_text),
                signature: thinking_signature.clone(),
                redacted: None,
            });
        } else if let Some(thinking) = response.thinking.as_mut() {
            if thinking.signature.is_none() {
                thinking.signature = thinking_signature;
            }
        }
        if response.tool_calls.is_empty() {
            response.tool_calls = pending_tool_calls
                .into_iter()
                .filter_map(|(index, pending)| pending.into_tool_call(index))
                .collect();
        }

        Ok(response)
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let client = self.client.clone();
        let config = self.config.clone();
        let request = Self::convert_request(request, true);
        let endpoint_url = Self::endpoint_url(&config.base_url);
        let request_debug = capture_debug_json(
            &format!("openai_codex stream request POST {}", endpoint_url),
            &request,
        );

        let stream = async_stream::stream! {
            let mut request_debug = request_debug;
            let auth = match auth::resolve_auth(&client, &config).await {
                Ok(auth) => auth,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let mut builder = client
                .post(&endpoint_url)
                .header("Authorization", format!("Bearer {}", auth.access_token))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .json(&request);

            if let Some(account_id) = auth.account_id {
                builder = builder.header("ChatGPT-Account-Id", account_id);
            }

            let response = builder.send().await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    yield Err(AiError::Http(error.to_string()));
                    return;
                }
            };

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let _ = capture_debug_text(
                    &format!("openai_codex stream response {} {}", status, endpoint_url),
                    body.clone(),
                );
                yield Err(api_error_from_response(status, &body));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut pending_tool_calls: HashMap<usize, PendingToolCallState> = HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Err(AiError::Http(error.to_string()));
                        return;
                    }
                };

                let chunk_str = match String::from_utf8(chunk.to_vec()) {
                    Ok(chunk_str) => chunk_str,
                    Err(_) => continue,
                };

                buffer.push_str(&chunk_str);

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer = buffer[pos + 1..].to_string();

                    let line = line.trim();
                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line[6..];
                    let response_debug =
                        capture_debug_text("openai_codex stream sse", line.to_string());
                    if data == "[DONE]" {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: None,
                            thinking_signature: None,
                            tool_call_deltas: Vec::new(),
                            done: true,
                            debug: if request_debug.is_some() || response_debug.is_some() {
                                Some(DebugTrace {
                                    request: request_debug.take(),
                                    response: response_debug,
                                })
                            } else {
                                None
                            },
                        });
                        return;
                    }

                    let event: OpenAiCodexEvent = match serde_json::from_str(data) {
                        Ok(event) => event,
                        Err(_) => continue,
                    };

                    match event.event_type.as_str() {
                        "response.output_text.delta" => {
                            let delta = event.delta.unwrap_or_default();
                            if delta.is_empty() {
                                continue;
                            }
                            yield Ok(StreamChunk {
                                delta,
                                thinking_delta: None,
                                thinking_signature: None,
                                tool_call_deltas: Vec::new(),
                                done: false,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug.clone(),
                                    })
                                } else {
                                    None
                                },
                            });
                        }
                        "response.output_item.added" => {
                            let Some(output_index) = event.output_index else {
                                continue;
                            };
                            let Some(item) = event.item else {
                                continue;
                            };
                            if item.item_type != "function_call" {
                                continue;
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
                                continue;
                            }

                            yield Ok(StreamChunk {
                                delta: String::new(),
                                thinking_delta: None,
                                thinking_signature: None,
                                tool_call_deltas: vec![ToolCallDelta {
                                    index: output_index,
                                    id,
                                    name,
                                    arguments: None,
                                }],
                                done: false,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug.clone(),
                                    })
                                } else {
                                    None
                                },
                            });
                        }
                        "response.function_call_arguments.delta" => {
                            let Some(output_index) = event.output_index else {
                                continue;
                            };
                            let Some(arguments) = event.delta else {
                                continue;
                            };

                            let pending = pending_tool_calls.entry(output_index).or_default();
                            pending.saw_argument_delta = true;
                            pending.arguments.push_str(&arguments);

                            yield Ok(StreamChunk {
                                delta: String::new(),
                                thinking_delta: None,
                                thinking_signature: None,
                                tool_call_deltas: vec![ToolCallDelta {
                                    index: output_index,
                                    id: None,
                                    name: None,
                                    arguments: Some(arguments),
                                }],
                                done: false,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug.clone(),
                                    })
                                } else {
                                    None
                                },
                            });
                        }
                        "response.output_item.done" => {
                            let Some(output_index) = event.output_index else {
                                continue;
                            };
                            let Some(item) = event.item else {
                                continue;
                            };

                            if item.item_type == "reasoning" {
                                if let Some(signature) = item.encrypted_content {
                                    yield Ok(StreamChunk {
                                        delta: String::new(),
                                        thinking_delta: None,
                                        thinking_signature: Some(signature),
                                        tool_call_deltas: Vec::new(),
                                        done: false,
                                        debug: if request_debug.is_some() || response_debug.is_some() {
                                            Some(DebugTrace {
                                                request: request_debug.take(),
                                                response: response_debug.clone(),
                                            })
                                        } else {
                                            None
                                        },
                                    });
                                }
                                continue;
                            }

                            if item.item_type != "function_call" {
                                continue;
                            }

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
                                continue;
                            }

                            yield Ok(StreamChunk {
                                delta: String::new(),
                                thinking_delta: None,
                                thinking_signature: None,
                                tool_call_deltas: vec![ToolCallDelta {
                                    index: output_index,
                                    id,
                                    name,
                                    arguments,
                                }],
                                done: false,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug.clone(),
                                    })
                                } else {
                                    None
                                },
                            });
                        }
                        "response.reasoning_summary_text.delta" => {
                            let thinking_delta = event.delta.or(event.text);
                            if thinking_delta.is_none() {
                                continue;
                            }
                            yield Ok(StreamChunk {
                                delta: String::new(),
                                thinking_delta,
                                thinking_signature: None,
                                tool_call_deltas: Vec::new(),
                                done: false,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug.clone(),
                                    })
                                } else {
                                    None
                                },
                            });
                        }
                        "response.completed" => {
                            yield Ok(StreamChunk {
                                delta: String::new(),
                                thinking_delta: None,
                                thinking_signature: None,
                                tool_call_deltas: Vec::new(),
                                done: true,
                                debug: if request_debug.is_some() || response_debug.is_some() {
                                    Some(DebugTrace {
                                        request: request_debug.take(),
                                        response: response_debug,
                                    })
                                } else {
                                    None
                                },
                            });
                            return;
                        }
                        _ => {}
                    }
                }
            }

            yield Ok(StreamChunk {
                delta: String::new(),
                thinking_delta: None,
                thinking_signature: None,
                tool_call_deltas: Vec::new(),
                done: true,
                debug: request_debug.map(|request| DebugTrace {
                    request: Some(request),
                    response: None,
                }),
            });
        };

        futures_util::StreamExt::boxed(stream)
    }

    fn config(&self) -> &AiConfig {
        &self.config
    }

    async fn list_models(&self) -> Result<Vec<String>, AiError> {
        Ok(vec![
            "gpt-5.1-codex-max".to_string(),
            "gpt-5.1-codex".to_string(),
            "gpt-5.1-codex-mini".to_string(),
            "gpt-5.2-codex".to_string(),
            "gpt-5.4".to_string(),
            "gpt-5.4-mini".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OpenAiCodexClient,
        auth::{base64_url_decode, extract_account_id_from_tokens},
        protocol::{OpenAiCodexInputItem, OpenAiCodexOutputItem, OpenAiCodexResponse},
    };
    use crate::ai::{
        ChatRequest, Message, ThinkingConfig, ThinkingEffort, ToolCall, ToolChoice, ToolDefinition,
    };
    use serde_json::json;

    #[test]
    fn decodes_base64_url_without_padding() {
        let decoded = base64_url_decode("eyJmb28iOiJiYXIifQ").expect("decoded");
        assert_eq!(
            String::from_utf8(decoded).expect("utf8"),
            r#"{"foo":"bar"}"#
        );
    }

    #[test]
    fn codex_request_uses_medium_reasoning_when_thinking_is_enabled() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_name: None,
                tool_result: None,
                tool_error: None,
            }],
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: Some(ThinkingConfig::enabled()),
        };

        let converted = OpenAiCodexClient::convert_request(request, false);
        assert_eq!(
            converted.reasoning.and_then(|reasoning| reasoning.effort),
            Some("medium")
        );
    }

    #[test]
    fn codex_request_uses_explicit_reasoning_effort() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_name: None,
                tool_result: None,
                tool_error: None,
            }],
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: Some(ThinkingConfig::enabled_with_effort(ThinkingEffort::XHigh)),
        };

        let converted = OpenAiCodexClient::convert_request(request, false);
        assert_eq!(
            converted.reasoning.and_then(|reasoning| reasoning.effort),
            Some("xhigh")
        );
    }

    #[test]
    fn codex_request_includes_tools_and_tool_outputs() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                Message::assistant_tool_calls(vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: json!({"city": "Tokyo"}),
                }]),
                Message::tool_result("call_123", "get_weather", json!({"temp_c": 21})),
            ],
            tools: vec![ToolDefinition::function(
                "get_weather",
                Some("Get weather".to_string()),
                json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            )],
            tool_choice: Some(ToolChoice::tool("get_weather")),
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: None,
        };

        let converted = OpenAiCodexClient::convert_request(request, true);

        assert_eq!(converted.tools.as_ref().map(Vec::len), Some(1));
        assert!(matches!(
            converted.tool_choice,
            Some(super::OpenAiCodexToolChoice::Function { .. })
        ));
        assert!(matches!(
            converted.input.first(),
            Some(OpenAiCodexInputItem::FunctionCall(_))
        ));
        assert!(matches!(
            converted.input.get(1),
            Some(OpenAiCodexInputItem::FunctionCallOutput(_))
        ));
    }

    #[test]
    fn codex_response_parses_tool_calls() {
        let response = OpenAiCodexResponse {
            id: "resp_123".to_string(),
            model: "gpt-5.4".to_string(),
            usage: None,
            output: vec![OpenAiCodexOutputItem {
                item_type: "function_call".to_string(),
                id: Some("fc_123".to_string()),
                call_id: Some("call_123".to_string()),
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
                status: Some("completed".to_string()),
                encrypted_content: None,
                content: Vec::new(),
                summary: Vec::new(),
                role: None,
            }],
        };

        let converted = OpenAiCodexClient::convert_response(response, None, None);
        assert_eq!(converted.tool_calls.len(), 1);
        assert_eq!(converted.tool_calls[0].id, "call_123");
        assert_eq!(converted.tool_calls[0].name, "get_weather");
        assert_eq!(converted.tool_calls[0].arguments, json!({"city": "Tokyo"}));
    }

    #[test]
    fn extracts_account_id_from_access_token_claims() {
        let token = "eyJhbGciOiJub25lIn0.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoib3JnXzEyMyJ9fQ.";
        assert_eq!(
            extract_account_id_from_tokens(None, Some(token)).as_deref(),
            Some("org_123")
        );
    }
}
