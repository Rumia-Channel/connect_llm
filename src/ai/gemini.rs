#![allow(dead_code)]

use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingOutput, Usage, capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    model_version: Option<String>,
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    thoughts_token_count: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiErrorEnvelope {
    error: GeminiErrorDetail,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiErrorDetail {
    #[serde(default)]
    code: Option<i32>,
    message: String,
    #[serde(default)]
    status: Option<String>,
}

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

    fn convert_role(role: &str) -> String {
        match role {
            "assistant" | "model" => "model".to_string(),
            _ => "user".to_string(),
        }
    }

    fn convert_message(message: super::Message) -> GeminiContent {
        let super::Message {
            role,
            content,
            thinking,
        } = message;

        let mut parts = Vec::new();

        if matches!(role.as_str(), "assistant" | "model") {
            if let Some(thinking) = &thinking {
                if let Some(text) = &thinking.text {
                    parts.push(GeminiPart {
                        text: Some(text.clone()),
                        thought: Some(true),
                        thought_signature: None,
                    });
                }
            }
        }

        let thought_signature = thinking.and_then(|thinking| thinking.signature);
        parts.push(GeminiPart {
            text: Some(content),
            thought: None,
            thought_signature,
        });

        GeminiContent {
            role: Some(Self::convert_role(&role)),
            parts,
        }
    }

    fn convert_system_instruction(system: String) -> GeminiContent {
        GeminiContent {
            role: Some("system".to_string()),
            parts: vec![GeminiPart {
                text: Some(system),
                thought: None,
                thought_signature: None,
            }],
        }
    }

    fn convert_request(request: ChatRequest) -> GeminiRequest {
        let ChatRequest {
            model: _,
            messages,
            max_tokens,
            temperature,
            system,
            thinking,
        } = request;

        GeminiRequest {
            contents: messages.into_iter().map(Self::convert_message).collect(),
            system_instruction: system.map(Self::convert_system_instruction),
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: max_tokens,
                temperature,
                thinking_config: thinking.and_then(|thinking| {
                    if !thinking.enabled {
                        return None;
                    }

                    Some(GeminiThinkingConfig {
                        include_thoughts: true,
                        thinking_budget: thinking.budget_tokens,
                    })
                }),
            }),
        }
    }

    fn parse_candidate(candidate: &GeminiCandidate) -> (String, ThinkingOutput) {
        let mut content = String::new();
        let mut thinking = ThinkingOutput::default();

        if let Some(candidate_content) = &candidate.content {
            for part in &candidate_content.parts {
                if let Some(signature) = &part.thought_signature {
                    thinking.signature = Some(signature.clone());
                }

                let Some(text) = &part.text else {
                    continue;
                };

                if part.thought.unwrap_or(false) {
                    match &mut thinking.text {
                        Some(existing) => existing.push_str(text),
                        None => thinking.text = Some(text.clone()),
                    }
                } else {
                    content.push_str(text);
                }
            }
        }

        (content, thinking)
    }

    fn convert_response(
        response: GeminiResponse,
        fallback_model: &str,
        request_debug: Option<String>,
        response_debug: Option<String>,
    ) -> ChatResponse {
        let candidate = response.candidates.first();
        let (content, thinking) = match candidate {
            Some(candidate) => Self::parse_candidate(candidate),
            None => (String::new(), ThinkingOutput::default()),
        };

        let thinking = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };

        let usage = response.usage_metadata.unwrap_or(GeminiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
            thoughts_token_count: None,
        });

        ChatResponse {
            id: response.response_id.unwrap_or_default(),
            content,
            model: response
                .model_version
                .unwrap_or_else(|| fallback_model.to_string()),
            usage: Usage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count
                    + usage.thoughts_token_count.unwrap_or(0),
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
impl AiClient for GeminiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let url = Self::generate_content_url(&self.config.base_url, &request.model);
        let request_model = request.model.clone();
        let gemini_request = Self::convert_request(request);
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
            if let Ok(error) = serde_json::from_str::<GeminiErrorEnvelope>(&body) {
                let mut parts = vec![format!("HTTP {}", status)];
                if let Some(code) = error.error.code {
                    parts.push(format!("code {}", code));
                }
                if let Some(kind) = error.error.status {
                    parts.push(kind);
                }
                return Err(AiError::Api(format!(
                    "{}: {}",
                    parts.join(" / "),
                    error.error.message
                )));
            }
            return Err(AiError::Api(format!("HTTP {}: {}", status, body)));
        }

        let gemini_response: GeminiResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(
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
        let gemini_request = Self::convert_request(request);
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
                if let Ok(error) = serde_json::from_str::<GeminiErrorEnvelope>(&body) {
                    let mut parts = vec![format!("HTTP {}", status)];
                    if let Some(code) = error.error.code {
                        parts.push(format!("code {}", code));
                    }
                    if let Some(kind) = error.error.status {
                        parts.push(kind);
                    }
                    yield Err(AiError::Api(format!("{}: {}", parts.join(" / "), error.error.message)));
                    return;
                }
                yield Err(AiError::Api(format!("HTTP {}: {}", status, body)));
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

                    let (content, thinking) = GeminiClient::parse_candidate(candidate);

                    if let Some(thinking_text) = thinking.text {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: Some(thinking_text),
                            thinking_signature: None,
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
            if let Ok(error) = serde_json::from_str::<GeminiErrorEnvelope>(&body) {
                let mut parts = vec![format!("HTTP {}", status)];
                if let Some(code) = error.error.code {
                    parts.push(format!("code {}", code));
                }
                if let Some(kind) = error.error.status {
                    parts.push(kind);
                }
                return Err(AiError::Api(format!(
                    "{}: {}",
                    parts.join(" / "),
                    error.error.message
                )));
            }
            return Err(AiError::Api(format!("HTTP {}: {}", status, body)));
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
