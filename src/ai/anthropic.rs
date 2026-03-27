#![allow(dead_code)]

use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, StreamChunk, ThinkingConfig,
    ThinkingDisplay, ThinkingOutput, Usage,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicRequestMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingRequest>,
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequestMessage {
    role: String,
    content: Vec<AnthropicRequestContentBlock>,
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequestContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicThinkingRequest {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    display: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContent>,
    model: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamResponse {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<AnthropicDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<AnthropicStreamMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamMessage {
    id: String,
    #[serde(rename = "type")]
    message_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

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

    fn convert_request_message(message: super::Message) -> AnthropicRequestMessage {
        let super::Message {
            role,
            content,
            thinking,
        } = message;

        let mut blocks = Vec::new();

        if let Some(thinking) = thinking {
            if thinking.text.is_some() || thinking.signature.is_some() {
                blocks.push(AnthropicRequestContentBlock {
                    content_type: "thinking".to_string(),
                    text: None,
                    thinking: Some(thinking.text.unwrap_or_default()),
                    signature: thinking.signature,
                    data: None,
                });
            }

            if let Some(redacted) = thinking.redacted {
                blocks.push(AnthropicRequestContentBlock {
                    content_type: "redacted_thinking".to_string(),
                    text: None,
                    thinking: None,
                    signature: None,
                    data: Some(redacted),
                });
            }
        }

        blocks.push(AnthropicRequestContentBlock {
            content_type: "text".to_string(),
            text: Some(content),
            thinking: None,
            signature: None,
            data: None,
        });

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
            max_tokens: max_tokens.unwrap_or(4096),
            system,
            temperature,
            stream: None,
            thinking: Self::convert_thinking_config(thinking.as_ref()),
        }
    }

    fn convert_response(response: AnthropicResponse) -> ChatResponse {
        let content = response
            .content
            .iter()
            .filter(|content| content.content_type == "text")
            .filter_map(|content| content.text.clone())
            .collect::<Vec<_>>()
            .join("");

        let mut thinking_output = ThinkingOutput::default();
        for content_block in &response.content {
            match content_block.content_type.as_str() {
                "thinking" => {
                    thinking_output.text = content_block.thinking.clone();
                    thinking_output.signature = content_block.signature.clone();
                }
                "redacted_thinking" => {
                    thinking_output.redacted = content_block.data.clone();
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
        }
    }
}

#[async_trait::async_trait]
impl AiClient for AnthropicClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let anthropic_request = Self::convert_request(request);

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

        if !status.is_success() {
            if let Ok(error) = serde_json::from_str::<AnthropicError>(&body) {
                return Err(AiError::Api(format!(
                    "{}: {}",
                    error.error.error_type, error.error.message
                )));
            }
            return Err(AiError::Api(format!("HTTP {}: {}", status, body)));
        }

        let anthropic_response: AnthropicResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(anthropic_response))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let mut anthropic_request = Self::convert_request(request);
        anthropic_request.stream = Some(true);
        let api_key = self.config.api_key.clone();

        let stream = async_stream::stream! {
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
                yield Err(AiError::Api(format!("HTTP {}: {}", status, body)));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut pending_event: Option<String> = None;
            let mut pending_data: Option<String> = None;

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

                    if line_trimmed.is_empty() {
                        if let (Some(event_type), Some(data)) =
                            (pending_event.take(), pending_data.take())
                        {
                            match event_type.as_str() {
                                "message_stop" => {
                                    yield Ok(StreamChunk {
                                        delta: String::new(),
                                        thinking_delta: None,
                                        thinking_signature: None,
                                        done: true,
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
                                                            done: false,
                                                        });
                                                    }
                                                }
                                                Some("thinking_delta") => {
                                                    if let Some(thinking) = delta.thinking {
                                                        yield Ok(StreamChunk {
                                                            delta: String::new(),
                                                            thinking_delta: Some(thinking),
                                                            thinking_signature: None,
                                                            done: false,
                                                        });
                                                    }
                                                }
                                                Some("signature_delta") => {
                                                    if let Some(signature) = delta.signature {
                                                        yield Ok(StreamChunk {
                                                            delta: String::new(),
                                                            thinking_delta: None,
                                                            thinking_signature: Some(signature),
                                                            done: false,
                                                        });
                                                    }
                                                }
                                                _ => {}
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
                done: true,
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
            return Ok(vec!["claude-3-5-sonnet-20241022".to_string()]);
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
