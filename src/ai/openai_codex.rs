#![allow(dead_code)]

mod auth;
mod convert;
mod protocol;
mod streaming;

use self::protocol::{
    OpenAiCodexRequest, OpenAiCodexResponse, PendingToolCallState, api_error_from_response,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    capture_debug_json, capture_debug_text,
};
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::HashMap;

pub use self::auth::{
    OpenAiCodexBrowserAuth, OpenAiCodexBrowserAuthOptions, login_openai_codex_via_browser,
    openai_codex_auth_path,
};

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
    fn convert_request(request: ChatRequest, stream: bool) -> OpenAiCodexRequest {
        convert::convert_request(request, stream)
    }

    fn convert_response(
        response: OpenAiCodexResponse,
        request_debug: Option<String>,
        response_debug: Option<String>,
    ) -> ChatResponse {
        convert::convert_response(response, request_debug, response_debug)
    }
}

#[async_trait::async_trait]
impl AiClient for OpenAiCodexClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = auth::resolve_auth(&self.client, &self.config).await?;
        let url = convert::endpoint_url(&self.config.base_url);
        let request = convert::convert_request(request, true);
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

        let parsed = streaming::parse_response_body(&body)?;
        let mut response =
            convert::convert_response(parsed.response, request_debug, response_debug);
        if response.content.is_empty() && !parsed.content.is_empty() {
            response.content = parsed.content;
        }
        streaming::finalize_response_thinking(
            &mut response.thinking,
            parsed.thinking_text,
            parsed.thinking_signature,
        );
        if response.tool_calls.is_empty() {
            response.tool_calls = parsed
                .pending_tool_calls
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
        let request = convert::convert_request(request, true);
        let endpoint_url = convert::endpoint_url(&config.base_url);
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

                    let chunk = match streaming::parse_stream_line(
                        &line,
                        &mut request_debug,
                        &mut pending_tool_calls,
                    ) {
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
        protocol::{
            OpenAiCodexInputItem, OpenAiCodexOutputItem, OpenAiCodexResponse, OpenAiCodexToolChoice,
        },
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
        assert_eq!(converted.temperature, None);
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
    fn codex_request_drops_temperature() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![Message::user("hello")],
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: Some(0.3),
            system: None,
            thinking: None,
        };

        let converted = OpenAiCodexClient::convert_request(request, false);
        assert_eq!(converted.temperature, None);
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
            Some(OpenAiCodexToolChoice::Function { .. })
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
