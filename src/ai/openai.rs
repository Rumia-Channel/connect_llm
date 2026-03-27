#![allow(dead_code)]

use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingConfig, ThinkingOutput, Usage, capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<OpenAiThinkingRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<OpenAiExtraBody>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiThinkingRequest {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    clear_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiExtraBody {
    google: OpenAiGoogleExtraBody,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiGoogleExtraBody {
    thinking_config: OpenAiGoogleThinkingConfig,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiGoogleThinkingConfig {
    include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiResponse {
    id: String,
    choices: Vec<OpenAiChoice>,
    model: String,
    usage: OpenAiUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessageResponse,
    #[serde(default)]
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiMessageResponse {
    role: String,
    content: String,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiStreamResponse {
    #[allow(dead_code)]
    id: String,
    choices: Vec<OpenAiStreamChoice>,
    #[allow(dead_code)]
    model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiError {
    error: OpenAiErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiErrorDetail {
    message: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
}

fn format_error_detail(
    status: reqwest::StatusCode,
    base_url: &str,
    detail: &OpenAiErrorDetail,
) -> String {
    if detail.code.as_deref() == Some("1113")
        || detail
            .message
            .contains("Insufficient balance or no resource package")
    {
        if base_url.contains("api.z.ai/api/coding/paas/v4") {
            return format!(
                "HTTP {} (code 1113): Z AI Coding plan quota is unavailable or this model is not supported on the coding endpoint. Use GLM-4.7 / GLM-4.6 / GLM-4.5 / GLM-4.5-Air, and check your Z AI Coding plan status.",
                status
            );
        }

        if base_url.contains("api.z.ai") {
            return format!(
                "HTTP {} (code 1113): Insufficient balance or no resource package. Recharge your Z AI balance for the general API endpoint, or use the separate Z AI Coding provider with the coding endpoint.",
                status
            );
        }
    }

    let mut parts = vec![format!("HTTP {}", status)];

    if let Some(code) = &detail.code {
        if !code.is_empty() {
            parts.push(format!("code {}", code));
        }
    }

    if let Some(error_type) = &detail.error_type {
        if !error_type.is_empty() {
            parts.push(error_type.clone());
        }
    }

    format!("{}: {}", parts.join(" / "), detail.message)
}

fn api_error_from_response(status: reqwest::StatusCode, body: &str, base_url: &str) -> AiError {
    if let Ok(error) = serde_json::from_str::<OpenAiError>(body) {
        return AiError::Api(format_error_detail(status, base_url, &error.error));
    }

    AiError::Api(format!("HTTP {}: {}", status, body))
}

pub struct OpenAiClient {
    client: Client,
    config: AiConfig,
}

impl OpenAiClient {
    pub fn new(config: AiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn normalized_base_url(base_url: &str) -> &str {
        base_url.trim_end_matches('/')
    }

    fn chat_completions_url(base_url: &str) -> String {
        let base_url = Self::normalized_base_url(base_url);
        if base_url.ends_with("/v1")
            || base_url.contains("/paas/v4")
            || base_url.ends_with("/openai")
        {
            format!("{}/chat/completions", base_url)
        } else {
            format!("{}/v1/chat/completions", base_url)
        }
    }

    fn models_url(base_url: &str) -> String {
        let base_url = Self::normalized_base_url(base_url);
        if base_url.ends_with("/v1")
            || base_url.contains("/paas/v4")
            || base_url.ends_with("/openai")
        {
            format!("{}/models", base_url)
        } else {
            format!("{}/v1/models", base_url)
        }
    }

    fn supports_reasoning_config(base_url: &str) -> bool {
        let base_url = Self::normalized_base_url(base_url);
        base_url.contains("api.moonshot.ai") || base_url.contains("api.z.ai")
    }

    fn is_google_openai_compat(base_url: &str) -> bool {
        Self::normalized_base_url(base_url).contains("generativelanguage.googleapis.com")
    }

    fn convert_thinking_config(
        base_url: &str,
        thinking: Option<&ThinkingConfig>,
    ) -> Option<OpenAiThinkingRequest> {
        let thinking = thinking?;
        if !Self::supports_reasoning_config(base_url) {
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
        if !thinking.enabled || !Self::is_google_openai_compat(base_url) {
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

    fn convert_request(request: ChatRequest, base_url: &str, stream: bool) -> OpenAiRequest {
        let ChatRequest {
            model,
            messages: request_messages,
            max_tokens,
            temperature,
            system,
            thinking,
        } = request;

        let mut messages = Vec::new();

        if let Some(system) = system {
            messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: system,
                reasoning_content: None,
            });
        }

        for message in request_messages {
            let super::Message {
                role,
                content,
                thinking,
            } = message;
            let reasoning_content = thinking.and_then(|thinking| thinking.text);
            messages.push(OpenAiMessage {
                role,
                content,
                reasoning_content,
            });
        }

        OpenAiRequest {
            model,
            messages,
            max_tokens,
            temperature,
            thinking: Self::convert_thinking_config(base_url, thinking.as_ref()),
            extra_body: Self::convert_google_extra_body(base_url, thinking.as_ref()),
            stream,
        }
    }

    fn convert_response(
        response: OpenAiResponse,
        request_debug: Option<String>,
        response_debug: Option<String>,
    ) -> ChatResponse {
        let thinking = response
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
                signature: None,
                redacted: None,
            });

        let content = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .unwrap_or_default();

        ChatResponse {
            id: response.id,
            content,
            model: response.model,
            usage: Usage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            },
            thinking,
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
impl AiClient for OpenAiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let url = Self::chat_completions_url(&self.config.base_url);
        let openai_request = Self::convert_request(request, &self.config.base_url, false);
        let request_debug =
            capture_debug_json(&format!("openai request POST {}", url), &openai_request);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let response_debug =
            capture_debug_text(&format!("openai response {} {}", status, url), body.clone());

        if !status.is_success() {
            return Err(api_error_from_response(
                status,
                &body,
                &self.config.base_url,
            ));
        }

        let openai_response: OpenAiResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(
            openai_response,
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let url = Self::chat_completions_url(&self.config.base_url);
        let openai_request = Self::convert_request(request, &self.config.base_url, true);
        let api_key = self.config.api_key.clone();
        let base_url = self.config.base_url.clone();
        let request_debug = capture_debug_json(
            &format!("openai stream request POST {}", url),
            &openai_request,
        );

        let stream = async_stream::stream! {
            let mut request_debug = request_debug;
            let response = reqwest::Client::new()
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&openai_request)
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
                    &format!("openai stream response {} {}", status, url),
                    body.clone(),
                );
                yield Err(api_error_from_response(status, &body, &base_url));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

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

                    let response_debug = capture_debug_text("openai stream sse", line.to_string());

                    let data = &line[6..];
                    if data == "[DONE]" {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: None,
                            thinking_signature: None,
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

                    let stream_response: OpenAiStreamResponse = match serde_json::from_str(data) {
                        Ok(response) => response,
                        Err(_) => continue,
                    };

                    if let Some(choice) = stream_response.choices.first() {
                        let delta = choice.delta.content.clone().unwrap_or_default();
                        let thinking_delta = choice
                            .delta
                            .reasoning_content
                            .clone()
                            .or_else(|| choice.delta.reasoning.clone());
                        let done = choice.finish_reason.is_some();

                        yield Ok(StreamChunk {
                            delta,
                            thinking_delta,
                            thinking_signature: None,
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
        list_models(&self.config.base_url, &self.config.api_key).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelInfo {
    id: String,
}

pub async fn list_models(base_url: &str, api_key: &str) -> Result<Vec<String>, AiError> {
    let url = OpenAiClient::models_url(base_url);
    let response = reqwest::Client::new()
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .map_err(|error| AiError::Http(error.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AiError::Http(error.to_string()))?;

    if !status.is_success() {
        if base_url.contains("api.z.ai/api/coding/paas/v4") {
            return Ok(vec![
                "glm-4.7".to_string(),
                "glm-4.6".to_string(),
                "glm-4.5".to_string(),
                "glm-4.5-air".to_string(),
            ]);
        }
        if base_url.contains("api.z.ai") {
            return Ok(vec![
                "glm-5".to_string(),
                "glm-4.7".to_string(),
                "glm-4.7-flash".to_string(),
                "glm-4.5-air".to_string(),
            ]);
        }
        return Err(api_error_from_response(status, &body, base_url));
    }

    let models: ModelsResponse =
        serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

    Ok(models.data.into_iter().map(|model| model.id).collect())
}
