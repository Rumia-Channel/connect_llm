#![allow(dead_code)]

mod auth;
mod convert;
mod protocol;

use self::protocol::{
    GitHubCopilotResponse, GitHubCopilotStreamResponse, api_error_from_response,
    parse_tool_call_deltas,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;

pub use self::auth::{
    GitHubCopilotDeviceAuth, GitHubCopilotDeviceAuthOptions, github_copilot_auth_path,
    login_github_copilot_via_device,
};

const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
pub struct GitHubCopilotClient {
    client: Client,
    config: AiConfig,
}

impl GitHubCopilotClient {
    fn fallback_model_ids() -> Vec<String> {
        vec![
            "claude-sonnet-4.6".to_string(),
            "claude-sonnet-4.5".to_string(),
            "gpt-4o".to_string(),
            "gpt-4.1".to_string(),
            "gpt-4.1-mini".to_string(),
            "gpt-4.1-nano".to_string(),
            "o1".to_string(),
            "o1-mini".to_string(),
            "o3-mini".to_string(),
        ]
    }

    pub fn new(config: AiConfig) -> Result<Self, AiError> {
        Ok(Self {
            client: config.http_client()?,
            config,
        })
    }
}

#[async_trait::async_trait]
impl AiClient for GitHubCopilotClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = auth::resolve_auth(&self.config).await?;
        let url = convert::chat_completions_url(&auth.base_url);
        let copilot_request = convert::convert_request(request, false);
        let initiator = convert::initiator_for_messages(&copilot_request.messages);
        let request_debug = capture_debug_json(
            &format!("github_copilot request POST {}", url),
            &copilot_request,
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.api_token))
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .header("Openai-Intent", "conversation-edits")
            .header("x-initiator", initiator)
            .json(&copilot_request)
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        let response_debug = capture_debug_text(
            &format!("github_copilot response {} {}", status, url),
            body.clone(),
        );

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let response: GitHubCopilotResponse =
            serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;

        Ok(convert::convert_response(
            response,
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let config = self.config.clone();
        let client = self.client.clone();
        let stream = async_stream::stream! {
            let auth = match auth::resolve_auth(&config).await {
                Ok(auth) => auth,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let url = convert::chat_completions_url(&auth.base_url);
            let copilot_request = convert::convert_request(request, true);
            let initiator = convert::initiator_for_messages(&copilot_request.messages);
            let mut request_debug = capture_debug_json(
                &format!("github_copilot stream request POST {}", url),
                &copilot_request,
            );

            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", auth.api_token))
                .header("Content-Type", "application/json")
                .header("User-Agent", USER_AGENT)
                .header("Openai-Intent", "conversation-edits")
                .header("x-initiator", initiator)
                .json(&copilot_request)
                .send()
                .await;

            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    yield Err(AiError::http(error.to_string()));
                    return;
                }
            };

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let _ = capture_debug_text(
                    &format!("github_copilot stream response {} {}", status, url),
                    body.clone(),
                );
                yield Err(api_error_from_response(status, &body));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Err(AiError::http(error.to_string()));
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

                    let response_debug =
                        capture_debug_text("github_copilot stream sse", line.to_string());
                    let data = &line[6..];

                    if data == "[DONE]" {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: None,
                            thinking_signature: None,
                            images: Vec::new(),
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

                    let stream_response: GitHubCopilotStreamResponse =
                        match serde_json::from_str(data) {
                            Ok(response) => response,
                            Err(_) => continue,
                        };

                    if let Some(choice) = stream_response.choices.first() {
                        let delta = choice.delta.content.clone().unwrap_or_default();
                        let thinking_delta = choice.delta.reasoning_text.clone();
                        let thinking_signature = choice.delta.reasoning_opaque.clone();
                        let tool_call_deltas =
                            parse_tool_call_deltas(choice.delta.tool_calls.clone());
                        let done = choice.finish_reason.is_some();

                        yield Ok(StreamChunk {
                            delta,
                            thinking_delta,
                            thinking_signature,
                            images: Vec::new(),
                            tool_call_deltas,
                            done,
                            debug: if request_debug.is_some() || response_debug.is_some() {
                                Some(DebugTrace {
                                    request: request_debug.take(),
                                    response: response_debug,
                                })
                            } else {
                                None
                            },
                        });

                        if done {
                            return;
                        }
                    }
                }
            }

            yield Ok(StreamChunk {
                delta: String::new(),
                thinking_delta: None,
                thinking_signature: None,
                images: Vec::new(),
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
        let auth = auth::resolve_auth(&self.config).await?;
        let url = convert::models_url(&auth.base_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", auth.api_token))
            .header("User-Agent", USER_AGENT)
            .header("Openai-Intent", "conversation-edits")
            .header("x-initiator", "user")
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        if !status.is_success() {
            return Ok(Self::fallback_model_ids());
        }

        #[derive(Debug, Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }

        #[derive(Debug, Deserialize)]
        struct ModelInfo {
            id: String,
        }

        let models: ModelsResponse = match serde_json::from_str(&body) {
            Ok(models) => models,
            Err(_) => return Ok(Self::fallback_model_ids()),
        };

        let model_ids: Vec<String> = models.data.into_iter().map(|model| model.id).collect();
        if model_ids.is_empty() {
            Ok(Self::fallback_model_ids())
        } else {
            Ok(model_ids)
        }
    }
}
