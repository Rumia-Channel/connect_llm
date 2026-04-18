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
    pub fn new(config: AiConfig) -> Result<Self, AiError> {
        Ok(Self {
            client: config.http_client()?,
            config,
        })
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
        let url = convert::endpoint_url(self.config.base_url());
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
            .map_err(|error| AiError::http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;
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
        let endpoint_url = convert::endpoint_url(config.base_url());
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
                    yield Err(AiError::http(error.to_string()));
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
