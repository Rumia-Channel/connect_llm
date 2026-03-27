#![allow(dead_code)]

mod convert;
mod protocol;

use self::protocol::{GeminiResponse, api_error_from_response};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk, ToolCallDelta,
    capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
pub struct GeminiClient {
    client: Client,
    config: AiConfig,
}

impl GeminiClient {
    pub fn new(config: AiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn normalized_base_url(base_url: &str) -> &str {
        base_url.trim_end_matches('/')
    }

    fn model_path(model: &str) -> String {
        if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{}", model)
        }
    }

    fn generate_content_url(base_url: &str, model: &str) -> String {
        format!(
            "{}/{}:generateContent",
            Self::normalized_base_url(base_url),
            Self::model_path(model)
        )
    }

    fn stream_generate_content_url(base_url: &str, model: &str) -> String {
        format!(
            "{}/{}:streamGenerateContent?alt=sse",
            Self::normalized_base_url(base_url),
            Self::model_path(model)
        )
    }

    fn models_url(base_url: &str) -> String {
        format!("{}/models", Self::normalized_base_url(base_url))
    }
}

#[async_trait::async_trait]
impl AiClient for GeminiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let url = Self::generate_content_url(&self.config.base_url, &request.model);
        let request_model = request.model.clone();
        let gemini_request = convert::convert_request(request);
        let request_debug =
            capture_debug_json(&format!("gemini request POST {}", url), &gemini_request);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;
        let response_debug =
            capture_debug_text(&format!("gemini response {} {}", status, url), body.clone());

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let gemini_response: GeminiResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(convert::convert_response(
            gemini_response,
            &request_model,
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let url = Self::stream_generate_content_url(&self.config.base_url, &request.model);
        let api_key = self.config.api_key.clone();
        let request_model = request.model.clone();
        let gemini_request = convert::convert_request(request);
        let request_debug = capture_debug_json(
            &format!("gemini stream request POST {}", url),
            &gemini_request,
        );

        let stream = async_stream::stream! {
            let mut request_debug = request_debug;
            let response = reqwest::Client::new()
                .post(&url)
                .header("x-goog-api-key", api_key)
                .header("Content-Type", "application/json")
                .json(&gemini_request)
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
                    &format!("gemini stream response {} {}", status, url),
                    body.clone(),
                );
                yield Err(api_error_from_response(status, &body));
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            while let Some(chunk_result) = byte_stream.next().await {
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
                    let response_debug = capture_debug_text("gemini stream sse", line.to_string());
                    let stream_response: GeminiResponse = match serde_json::from_str(data) {
                        Ok(response) => response,
                        Err(_) => continue,
                    };

                    let Some(candidate) = stream_response.candidates.first() else {
                        continue;
                    };

                    let (content, thinking, tool_calls) = convert::parse_candidate(candidate);

                    if let Some(thinking_text) = thinking.text {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: Some(thinking_text),
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

                    if let Some(signature) = thinking.signature {
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

                    if !tool_calls.is_empty() {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: None,
                            thinking_signature: None,
                            tool_call_deltas: tool_calls
                                .into_iter()
                                .enumerate()
                                .map(|(index, tool_call)| ToolCallDelta {
                                    index,
                                    id: Some(tool_call.id),
                                    name: Some(tool_call.name),
                                    arguments: Some(tool_call.arguments.to_string()),
                                })
                                .collect(),
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

                    if !content.is_empty() {
                        yield Ok(StreamChunk {
                            delta: content,
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

                    if candidate.finish_reason.is_some() {
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
                }
            }

            let _ = request_model;
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
        #[derive(Debug, Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct GeminiModelsResponse {
            #[serde(default)]
            models: Vec<GeminiModel>,
        }

        #[derive(Debug, Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct GeminiModel {
            #[serde(default)]
            name: String,
            #[serde(default)]
            base_model_id: Option<String>,
            #[serde(default)]
            supported_generation_methods: Vec<String>,
        }

        let url = Self::models_url(&self.config.base_url);
        let response = reqwest::Client::new()
            .get(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let models: GeminiModelsResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(models
            .models
            .into_iter()
            .filter(|model| {
                model
                    .supported_generation_methods
                    .iter()
                    .any(|method| method == "generateContent")
            })
            .map(|model| {
                model.base_model_id.unwrap_or_else(|| {
                    model
                        .name
                        .strip_prefix("models/")
                        .unwrap_or(&model.name)
                        .to_string()
                })
            })
            .collect())
    }
}
