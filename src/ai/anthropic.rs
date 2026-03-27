#![allow(dead_code)]

mod protocol;

use self::protocol::{
    AnthropicRequest, AnthropicRequestContentBlock, AnthropicRequestMessage, AnthropicResponse,
    AnthropicStreamResponse, AnthropicThinkingRequest, AnthropicToolChoice,
    AnthropicToolDefinition, api_error_from_response,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingConfig, ThinkingDisplay, ThinkingOutput, ToolCall, ToolCallDelta, ToolChoice,
    ToolDefinition, Usage, capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
pub struct AnthropicClient {
    client: Client,
    config: AiConfig,
}

impl AnthropicClient {
    pub fn new(config: AiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<AnthropicToolDefinition>> {
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

    fn convert_tool_choice(choice: Option<&ToolChoice>) -> Option<AnthropicToolChoice> {
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

    fn convert_request_message(message: super::Message) -> AnthropicRequestMessage {
        let super::Message {
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

    fn convert_thinking_config(
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

    fn convert_request(request: ChatRequest) -> AnthropicRequest {
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

        let messages = messages
            .into_iter()
            .map(Self::convert_request_message)
            .collect();

        AnthropicRequest {
            model,
            messages,
            tools: Self::convert_tools(&tools),
            tool_choice: Self::convert_tool_choice(tool_choice.as_ref()),
            max_tokens: max_tokens.unwrap_or(4096),
            system,
            temperature,
            stream: None,
            thinking: Self::convert_thinking_config(thinking.as_ref()),
        }
    }

    fn convert_response(
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
}

#[async_trait::async_trait]
impl AiClient for AnthropicClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let anthropic_request = Self::convert_request(request);
        let request_debug = capture_debug_json(
            &format!("anthropic request POST {}", url),
            &anthropic_request,
        );

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;
        let response_debug = capture_debug_text(
            &format!("anthropic response {} {}", status, url),
            body.clone(),
        );

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let anthropic_response: AnthropicResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(
            anthropic_response,
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let mut anthropic_request = Self::convert_request(request);
        anthropic_request.stream = Some(true);
        let api_key = self.config.api_key.clone();
        let request_debug = capture_debug_json(
            &format!("anthropic stream request POST {}", url),
            &anthropic_request,
        );

        let stream = async_stream::stream! {
            let mut request_debug = request_debug;
            let response = reqwest::Client::new()
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&anthropic_request)
                .send()
                .await;

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
                    &format!("anthropic stream response {} {}", status, url),
                    body.clone(),
                );
                yield Err(AiError::Api(format!("HTTP {}: {}", status, body)));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut pending_event: Option<String> = None;
            let mut pending_data: Option<String> = None;
            let mut pending_debug_lines: Vec<String> = Vec::new();

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

                    let line_trimmed = line.trim_end_matches('\r').trim_end();
                    if !line_trimmed.is_empty() {
                        let _ = capture_debug_text("anthropic stream sse", line_trimmed.to_string());
                        pending_debug_lines.push(line_trimmed.to_string());
                    }

                    if line_trimmed.is_empty() {
                        let response_debug = if pending_debug_lines.is_empty() {
                            None
                        } else {
                            Some(pending_debug_lines.join("\n"))
                        };
                        pending_debug_lines.clear();

                        if let (Some(event_type), Some(data)) =
                            (pending_event.take(), pending_data.take())
                        {
                            match event_type.as_str() {
                                "message_stop" => {
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
                                "error" => {
                                    yield Err(AiError::Api(data));
                                    return;
                                }
                                "content_block_delta" => {
                                    if let Ok(stream_response) =
                                        serde_json::from_str::<AnthropicStreamResponse>(&data)
                                    {
                                        if let Some(delta) = stream_response.delta {
                                            match delta.delta_type.as_deref() {
                                                Some("text_delta") => {
                                                    if let Some(text) = delta.text {
                                                        yield Ok(StreamChunk {
                                                            delta: text,
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
                                                }
                                                Some("thinking_delta") => {
                                                    if let Some(thinking) = delta.thinking {
                                                        yield Ok(StreamChunk {
                                                            delta: String::new(),
                                                            thinking_delta: Some(thinking),
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
                                                }
                                                Some("signature_delta") => {
                                                    if let Some(signature) = delta.signature {
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
                                                }
                                                Some("input_json_delta") => {
                                                    if let Some(arguments) = delta.partial_json {
                                                        yield Ok(StreamChunk {
                                                            delta: String::new(),
                                                            thinking_delta: None,
                                                            thinking_signature: None,
                                                            tool_call_deltas: vec![ToolCallDelta {
                                                                index: stream_response.index.unwrap_or(0),
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
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                "content_block_start" => {
                                    if let Ok(stream_response) =
                                        serde_json::from_str::<AnthropicStreamResponse>(&data)
                                    {
                                        if let Some(content_block) = stream_response.content_block {
                                            if content_block.content_type == "tool_use" {
                                                yield Ok(StreamChunk {
                                                    delta: String::new(),
                                                    thinking_delta: None,
                                                    thinking_signature: None,
                                                    tool_call_deltas: vec![ToolCallDelta {
                                                        index: stream_response.index.unwrap_or(0),
                                                        id: content_block.id,
                                                        name: content_block.name,
                                                        arguments: content_block
                                                            .input
                                                            .map(|input| input.to_string()),
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
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        continue;
                    }

                    if line_trimmed.starts_with("event:") {
                        let event = line_trimmed[6..].trim_start().to_string();
                        pending_event = Some(event);
                    } else if line_trimmed.starts_with("data:") {
                        let data = line_trimmed[5..].trim_start().to_string();
                        pending_data = Some(data);
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

        stream.boxed()
    }

    fn config(&self) -> &AiConfig {
        &self.config
    }

    async fn list_models(&self) -> Result<Vec<String>, AiError> {
        let url = format!("{}/v1/models", self.config.base_url);

        let response = reqwest::Client::new()
            .get(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            if self.config.base_url.contains("kimi.com/coding") {
                return Ok(vec![
                    "kimi-for-coding".to_string(),
                    "anthropic/k2p5".to_string(),
                ]);
            }
            return Ok(vec!["claude-sonnet-4-20250514".to_string()]);
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelData>,
        }

        #[derive(Deserialize)]
        struct ModelData {
            id: String,
        }

        let models: ModelsResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(models.data.into_iter().map(|model| model.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::AnthropicClient;
    use crate::ai::{ChatRequest, Message, ToolCall, ToolChoice, ToolDefinition};
    use serde_json::json;

    #[test]
    fn convert_request_maps_tools_and_tool_results() {
        let mut request = ChatRequest::new(
            "claude-sonnet-4-20250514",
            vec![
                Message::assistant_tool_calls(vec![ToolCall {
                    id: "toolu_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: json!({"city": "Tokyo"}),
                }]),
                Message::tool_result("toolu_1", "get_weather", json!({"temperature_c": 22})),
            ],
        );
        request.tools = vec![ToolDefinition::function(
            "get_weather",
            Some("Return weather".to_string()),
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )];
        request.tool_choice = Some(ToolChoice::Tool("get_weather".to_string()));

        let converted = AnthropicClient::convert_request(request);
        assert_eq!(converted.tools.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            converted
                .tool_choice
                .as_ref()
                .and_then(|choice| choice.name.as_deref()),
            Some("get_weather")
        );
        assert_eq!(converted.messages[0].content[0].content_type, "tool_use");
        assert_eq!(converted.messages[1].role, "user");
        assert_eq!(converted.messages[1].content[0].content_type, "tool_result");
    }
}
