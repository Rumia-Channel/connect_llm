#![allow(dead_code)]

mod convert;
mod protocol;
mod streaming;

use self::protocol::{AnthropicResponse, api_error_from_response};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
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

    fn is_native_anthropic(&self) -> bool {
        self.config.base_url.contains("api.anthropic.com")
    }

    fn is_kimi_coding(&self) -> bool {
        self.config.base_url.contains("kimi.com/coding")
    }

    fn anthropic_beta_header(&self) -> Option<&'static str> {
        self.is_native_anthropic()
            .then_some("interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14")
    }

    fn default_max_tokens(&self, model: &str) -> u32 {
        if self.is_kimi_coding() {
            return 32_768;
        }

        if self.is_native_anthropic() {
            return match model {
                model if model.starts_with("claude-opus-4-1") => 32_000,
                model if model.starts_with("claude-opus-4") => 32_000,
                model if model.starts_with("claude-sonnet-4") => 64_000,
                model if model.starts_with("claude-3-7-sonnet") => 64_000,
                model if model.starts_with("claude-3-5-sonnet") => 8_192,
                model if model.starts_with("claude-3-5-haiku") => 8_192,
                model if model.starts_with("claude-3-haiku") => 4_096,
                _ => 8_192,
            };
        }

        8_192
    }

    fn default_thinking_budget(&self, request: &ChatRequest) -> Option<u32> {
        let thinking = request.thinking.as_ref()?;
        if !thinking.enabled || thinking.budget_tokens.is_some() {
            return None;
        }

        if self.is_kimi_coding() {
            let max_tokens = request
                .max_tokens
                .unwrap_or_else(|| self.default_max_tokens(&request.model));
            return Some((max_tokens / 2).saturating_sub(1).min(16_000));
        }

        None
    }

    fn resolve_request_defaults(&self, mut request: ChatRequest) -> ChatRequest {
        if request.max_tokens.is_none() {
            request.max_tokens = Some(self.default_max_tokens(&request.model));
        }

        let default_thinking_budget = self.default_thinking_budget(&request);
        if let (Some(thinking), Some(budget_tokens)) =
            (&mut request.thinking, default_thinking_budget)
        {
            thinking.budget_tokens = Some(budget_tokens);
        }

        request
    }
}

#[async_trait::async_trait]
impl AiClient for AnthropicClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let request = self.resolve_request_defaults(request);
        let url = format!("{}/v1/messages", self.config.base_url);
        let anthropic_request = convert::convert_request(request);
        let request_debug = capture_debug_json(
            &format!("anthropic request POST {}", url),
            &anthropic_request,
        );

        let mut builder = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");
        if let Some(beta) = self.anthropic_beta_header() {
            builder = builder.header("anthropic-beta", beta);
        }

        let response = builder
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

        Ok(convert::convert_response(
            anthropic_response,
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let request = self.resolve_request_defaults(request);
        let url = format!("{}/v1/messages", self.config.base_url);
        let mut anthropic_request = convert::convert_request(request);
        anthropic_request.stream = Some(true);
        let api_key = self.config.api_key.clone();
        let anthropic_beta_header = self.anthropic_beta_header().map(str::to_string);
        let request_debug = capture_debug_json(
            &format!("anthropic stream request POST {}", url),
            &anthropic_request,
        );

        let stream = async_stream::stream! {
            let mut request_debug = request_debug;
            let mut builder = reqwest::Client::new()
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json");
            if let Some(beta) = anthropic_beta_header.as_deref() {
                builder = builder.header("anthropic-beta", beta);
            }

            let response = builder
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
            let mut sse_state = streaming::AnthropicSseState::default();

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

                    let chunk = match streaming::consume_line(&line, &mut sse_state, &mut request_debug) {
                        Ok(Some(chunk)) => chunk,
                        Ok(None) => continue,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    };

                    let done = chunk.done;
                    yield Ok(chunk);
                    if done {
                        return;
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
        let url = format!("{}/v1/models", self.config.base_url);

        let mut builder = reqwest::Client::new()
            .get(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01");
        if let Some(beta) = self.anthropic_beta_header() {
            builder = builder.header("anthropic-beta", beta);
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

        if !status.is_success() {
            if self.is_kimi_coding() {
                return Ok(vec!["k2p5".to_string(), "kimi-k2-thinking".to_string()]);
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
    use super::{AnthropicClient, convert};
    use crate::ai::{
        AiConfig, ChatRequest, Message, ThinkingConfig, ToolCall, ToolChoice, ToolDefinition,
    };
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

        let converted = convert::convert_request(request);
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

    #[test]
    fn resolves_anthropic_default_max_tokens() {
        let client = AnthropicClient::new(AiConfig {
            api_key: "test".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
        });

        let request = ChatRequest::new("claude-sonnet-4-20250514", vec![Message::user("hi")]);
        let resolved = client.resolve_request_defaults(request);
        assert_eq!(resolved.max_tokens, Some(64_000));
    }

    #[test]
    fn resolves_kimi_coding_defaults() {
        let client = AnthropicClient::new(AiConfig {
            api_key: "test".to_string(),
            base_url: "https://api.kimi.com/coding".to_string(),
            model: "k2p5".to_string(),
        });

        let mut request = ChatRequest::new("k2p5", vec![Message::user("hi")]);
        request.thinking = Some(ThinkingConfig::enabled());

        let resolved = client.resolve_request_defaults(request);
        assert_eq!(resolved.max_tokens, Some(32_768));
        assert_eq!(
            resolved
                .thinking
                .and_then(|thinking| thinking.budget_tokens),
            Some(16_000)
        );
    }
}
