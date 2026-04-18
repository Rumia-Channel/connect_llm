#![allow(dead_code)]

mod convert;
mod protocol;

use self::protocol::{
    ModelsResponse, OpenAiResponse, OpenAiStreamResponse, api_error_from_response,
    convert_tool_call_deltas,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;

pub struct OpenAiClient {
    client: Client,
    config: AiConfig,
}

impl OpenAiClient {
    pub fn new(config: AiConfig) -> Result<Self, AiError> {
        Ok(Self {
            client: config.http_client()?,
            config,
        })
    }
}

#[async_trait::async_trait]
impl AiClient for OpenAiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let api_key = self.config.require_bearer_token("chat")?;
        let url = convert::chat_completions_url(self.config.base_url());
        let openai_request = convert::convert_request(request, self.config.base_url(), false);
        let request_debug =
            capture_debug_json(&format!("openai request POST {}", url), &openai_request);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;

        let response_debug =
            capture_debug_text(&format!("openai response {} {}", status, url), body.clone());

        if !status.is_success() {
            return Err(api_error_from_response(
                status,
                &body,
                self.config.base_url(),
            ));
        }

        let openai_response: OpenAiResponse =
            serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;

        Ok(convert::convert_response(
            openai_response,
            self.config.base_url(),
            request_debug,
            response_debug,
        ))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let url = convert::chat_completions_url(self.config.base_url());
        let openai_request = convert::convert_request(request, self.config.base_url(), true);
        let api_key = self
            .config
            .require_bearer_token("chat_stream")
            .map(str::to_string);
        let base_url = self.config.base_url().to_string();
        let client = self.client.clone();
        let request_debug = capture_debug_json(
            &format!("openai stream request POST {}", url),
            &openai_request,
        );
        let is_google_openai_compat =
            convert::normalized_base_url(&base_url).contains("generativelanguage.googleapis.com");

        let stream = async_stream::stream! {
            let api_key = match api_key {
                Ok(api_key) => api_key,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let mut request_debug = request_debug;
            let mut thought_state = convert::EmbeddedThoughtState::default();
            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&openai_request)
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

                    let response_debug = capture_debug_text("openai stream sse", line.to_string());

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

                    let stream_response: OpenAiStreamResponse = match serde_json::from_str(data) {
                        Ok(response) => response,
                        Err(_) => continue,
                    };

                    if let Some(choice) = stream_response.choices.first() {
                        let raw_delta = choice.delta.content.clone().unwrap_or_default();
                        let google_extra = choice
                            .delta
                            .extra_content
                            .as_ref()
                            .and_then(|extra| extra.google.as_ref());
                        let google_thought = google_extra
                            .and_then(|google| google.thought)
                            .unwrap_or(false);
                        let google_thought_signature = google_extra
                            .and_then(|google| google.thought_signature.clone());
                        let (delta, embedded_thinking_delta) = if is_google_openai_compat {
                            thought_state.push_google_chunk(&raw_delta, google_thought)
                        } else {
                            (raw_delta, None)
                        };
                        let thinking_delta = choice
                            .delta
                            .reasoning_content
                            .clone()
                            .or_else(|| choice.delta.reasoning.clone())
                            .or(embedded_thinking_delta);

                        let tool_call_deltas =
                            convert_tool_call_deltas(choice.delta.tool_calls.clone());
                        let done = choice.finish_reason.is_some();

                        yield Ok(StreamChunk {
                            delta,
                            thinking_delta,
                            thinking_signature: google_thought_signature,
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
                            let (tail_delta, tail_thinking) = thought_state.clone().finish();
                            if !tail_delta.is_empty() || tail_thinking.is_some() {
                                yield Ok(StreamChunk {
                                    delta: tail_delta,
                                    thinking_delta: tail_thinking,
                                    thinking_signature: None,
                                    images: Vec::new(),
                                    tool_call_deltas: Vec::new(),
                                    done: false,
                                    debug: None,
                                });
                            }
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
        list_models(&self.client, &self.config).await
    }
}

pub async fn list_models(client: &Client, config: &AiConfig) -> Result<Vec<String>, AiError> {
    let api_key = config.require_bearer_token("list_models")?;
    let url = convert::models_url(config.base_url());
    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .map_err(|error| AiError::http(error.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AiError::http(error.to_string()))?;

    if !status.is_success() {
        if config.base_url().contains("api.z.ai/api/coding/paas/v4") {
            return Ok(vec![
                "glm-5".to_string(),
                "glm-4.7".to_string(),
                "glm-4.6".to_string(),
                "glm-4.5".to_string(),
                "glm-4.5-air".to_string(),
            ]);
        }
        if config.base_url().contains("api.z.ai") {
            return Ok(vec![
                "glm-5".to_string(),
                "glm-4.7".to_string(),
                "glm-4.7-flash".to_string(),
                "glm-4.5-air".to_string(),
            ]);
        }
        return Err(api_error_from_response(status, &body, config.base_url()));
    }

    let models: ModelsResponse =
        serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;

    Ok(models.data.into_iter().map(|model| model.id).collect())
}

#[cfg(test)]
mod tests {
    use super::OpenAiClient;
    use super::{
        convert,
        protocol::{
            OpenAiChoice, OpenAiMessageResponse, OpenAiResponse, OpenAiToolCall,
            OpenAiToolFunction, OpenAiUsage,
        },
    };
    use crate::ai::{
        AiAuth, AiConfig, AiProvider, ChatRequest, Message, ToolCall, ToolChoice, ToolDefinition,
    };
    use reqwest::Client;
    use serde_json::json;

    #[test]
    fn convert_request_includes_tools_and_tool_results() {
        let mut request = ChatRequest::new(
            "gpt-5.4",
            vec![
                Message::assistant_tool_calls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: json!({"city": "Tokyo"}),
                }]),
                Message::tool_result("call_1", "get_weather", json!({"temperature_c": 22})),
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
        request.tool_choice = Some(ToolChoice::Auto);

        let converted = convert::convert_request(request, "https://api.openai.com", false);
        assert_eq!(converted.tools.as_ref().map(Vec::len), Some(1));
        assert_eq!(converted.tool_choice, Some(json!("auto")));
        assert_eq!(
            converted.messages[0].tool_calls.as_ref().map(Vec::len),
            Some(1)
        );
        assert_eq!(
            converted.messages[1].tool_call_id.as_deref(),
            Some("call_1")
        );
        assert_eq!(
            converted.messages[1].content.as_deref(),
            Some("{\"temperature_c\":22}")
        );
        assert!(converted.messages[1].tool_calls.is_none());
    }

    #[test]
    fn convert_request_drops_temperature_for_openai_gpt5_models() {
        let mut request = ChatRequest::new("gpt-5.4", vec![Message::user("hello")]);
        request.temperature = Some(0.2);

        let converted = convert::convert_request(request, "https://api.openai.com", false);
        assert_eq!(converted.temperature, None);
    }

    #[test]
    fn convert_request_keeps_temperature_for_kimi_models() {
        let mut request = ChatRequest::new("kimi-k2.5", vec![Message::user("hello")]);
        request.temperature = Some(0.6);

        let converted = convert::convert_request(request, "https://api.moonshot.ai", false);
        assert_eq!(converted.temperature, Some(0.6));
    }

    #[test]
    fn convert_response_parses_tool_calls() {
        let response = OpenAiResponse {
            id: "resp_1".to_string(),
            choices: vec![OpenAiChoice {
                message: OpenAiMessageResponse {
                    role: "assistant".to_string(),
                    content: Some(String::new()),
                    reasoning_content: None,
                    reasoning: None,
                    extra_content: None,
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: OpenAiToolFunction {
                            name: "get_weather".to_string(),
                            arguments: "{\"city\":\"Tokyo\"}".to_string(),
                        },
                    }]),
                },
                finish_reason: "tool_calls".to_string(),
            }],
            model: "gpt-5.4".to_string(),
            usage: OpenAiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let converted = convert::convert_response(response, "https://api.openai.com", None, None);
        assert_eq!(converted.tool_calls.len(), 1);
        assert_eq!(converted.tool_calls[0].name, "get_weather");
        assert_eq!(converted.tool_calls[0].arguments, json!({"city": "Tokyo"}));
    }

    #[test]
    fn constructor_uses_injected_http_client() {
        let injected = Client::new();
        let config = AiConfig::new(AiProvider::OpenAi)
            .with_auth(AiAuth::BearerToken("test".to_string()))
            .with_http_client(injected.clone());
        let client = OpenAiClient::new(config).expect("client");
        let _ = client;
    }
}
