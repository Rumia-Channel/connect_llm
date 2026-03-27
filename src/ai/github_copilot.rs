#![allow(dead_code)]

mod auth;
mod protocol;

use self::protocol::{
    GitHubCopilotFunctionDefinition, GitHubCopilotMessage, GitHubCopilotRequest,
    GitHubCopilotResponse, GitHubCopilotStreamResponse, GitHubCopilotToolCall,
    GitHubCopilotToolDefinition, GitHubCopilotToolFunction, api_error_from_response,
    parse_tool_call_deltas, parse_tool_calls,
};
use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingEffort, ThinkingOutput, ToolCall, ToolChoice, ToolDefinition, Usage,
    capture_debug_json, capture_debug_text, serialize_tool_arguments,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

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
        if base_url.ends_with("/v1") {
            format!("{}/chat/completions", base_url)
        } else {
            format!("{}/chat/completions", base_url)
        }
    }

    fn models_url(base_url: &str) -> String {
        let base_url = Self::normalized_base_url(base_url);
        if base_url.ends_with("/v1") {
            format!("{}/models", base_url)
        } else {
            format!("{}/models", base_url)
        }
    }

    fn convert_effort(effort: ThinkingEffort) -> &'static str {
        match effort {
            ThinkingEffort::Minimal => "minimal",
            ThinkingEffort::Low => "low",
            ThinkingEffort::Medium => "medium",
            ThinkingEffort::High => "high",
            ThinkingEffort::XHigh => "xhigh",
        }
    }

    fn initiator_for_messages(messages: &[GitHubCopilotMessage]) -> &'static str {
        if messages
            .last()
            .map(|message| message.role.as_str() != "user")
            .unwrap_or(false)
        {
            "agent"
        } else {
            "user"
        }
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Option<Vec<GitHubCopilotToolDefinition>> {
        if tools.is_empty() {
            return None;
        }

        Some(
            tools
                .iter()
                .map(|tool| GitHubCopilotToolDefinition {
                    tool_type: "function",
                    function: GitHubCopilotFunctionDefinition {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.input_schema.clone(),
                    },
                })
                .collect(),
        )
    }

    fn convert_tool_choice(choice: Option<&ToolChoice>) -> Option<Value> {
        match choice? {
            ToolChoice::Auto => Some(json!("auto")),
            ToolChoice::None => Some(json!("none")),
            ToolChoice::Required => Some(json!("required")),
            ToolChoice::Tool(name) => Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                }
            })),
        }
    }

    fn convert_tool_calls(tool_calls: Vec<ToolCall>) -> Option<Vec<GitHubCopilotToolCall>> {
        if tool_calls.is_empty() {
            return None;
        }

        Some(
            tool_calls
                .into_iter()
                .map(|tool_call| GitHubCopilotToolCall {
                    id: tool_call.id,
                    call_type: "function".to_string(),
                    function: GitHubCopilotToolFunction {
                        name: tool_call.name,
                        arguments: serialize_tool_arguments(&tool_call.arguments),
                    },
                })
                .collect(),
        )
    }

    fn convert_request(request: ChatRequest, stream: bool) -> GitHubCopilotRequest {
        let ChatRequest {
            model,
            messages: request_messages,
            tools,
            tool_choice,
            max_tokens,
            temperature,
            system,
            thinking,
        } = request;

        let mut messages = Vec::new();

        if let Some(system) = system {
            messages.push(GitHubCopilotMessage {
                role: "system".to_string(),
                content: Some(system),
                reasoning_text: None,
                reasoning_opaque: None,
                tool_call_id: None,
                tool_calls: None,
            });
        }

        for message in request_messages {
            let super::Message {
                role,
                content,
                thinking,
                tool_calls,
                tool_call_id,
                tool_name: _,
                tool_result: _,
                tool_error: _,
            } = message;

            let (reasoning_text, reasoning_opaque) = match thinking {
                Some(thinking) => (thinking.text, thinking.signature.or(thinking.redacted)),
                None => (None, None),
            };

            let content = if role == "assistant" && content.is_empty() {
                None
            } else {
                Some(content)
            };

            messages.push(GitHubCopilotMessage {
                role,
                content,
                reasoning_text,
                reasoning_opaque,
                tool_call_id,
                tool_calls: Self::convert_tool_calls(tool_calls),
            });
        }

        let reasoning_effort = thinking
            .as_ref()
            .and_then(|thinking| thinking.effort)
            .map(Self::convert_effort);
        let thinking_budget = thinking
            .as_ref()
            .and_then(|thinking| thinking.enabled.then_some(thinking.budget_tokens))
            .flatten();

        GitHubCopilotRequest {
            model,
            messages,
            tools: Self::convert_tools(&tools),
            tool_choice: Self::convert_tool_choice(tool_choice.as_ref()),
            max_tokens,
            temperature,
            reasoning_effort,
            thinking_budget,
            stream,
        }
    }

    fn convert_response(
        response: GitHubCopilotResponse,
        request_debug: Option<String>,
        response_debug: Option<String>,
    ) -> ChatResponse {
        let message = response.choices.first().map(|choice| &choice.message);
        let content = message
            .and_then(|message| message.content.clone())
            .unwrap_or_default();
        let thinking = message.and_then(|message| {
            let thinking = ThinkingOutput {
                text: message.reasoning_text.clone(),
                signature: message.reasoning_opaque.clone(),
                redacted: None,
            };
            (!thinking.is_empty()).then_some(thinking)
        });
        let tool_calls = message
            .map(|message| parse_tool_calls(message.tool_calls.clone()))
            .unwrap_or_default();
        let usage = response.usage.unwrap_or_default();

        ChatResponse {
            id: response.id,
            content,
            model: response.model,
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
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
impl AiClient for GitHubCopilotClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = auth::resolve_auth(&self.config).await?;
        let url = Self::chat_completions_url(&auth.base_url);
        let copilot_request = Self::convert_request(request, false);
        let initiator = Self::initiator_for_messages(&copilot_request.messages);
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
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let response_debug = capture_debug_text(
            &format!("github_copilot response {} {}", status, url),
            body.clone(),
        );

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let response: GitHubCopilotResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(
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
        let stream = async_stream::stream! {
            let auth = match auth::resolve_auth(&config).await {
                Ok(auth) => auth,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let url = GitHubCopilotClient::chat_completions_url(&auth.base_url);
            let copilot_request = GitHubCopilotClient::convert_request(request, true);
            let initiator = GitHubCopilotClient::initiator_for_messages(&copilot_request.messages);
            let mut request_debug = capture_debug_json(
                &format!("github_copilot stream request POST {}", url),
                &copilot_request,
            );

            let response = reqwest::Client::new()
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
                    yield Err(AiError::Http(error.to_string()));
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

                    let response_debug =
                        capture_debug_text("github_copilot stream sse", line.to_string());
                    let data = &line[6..];

                    if data == "[DONE]" {
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
        let url = Self::models_url(&auth.base_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", auth.api_token))
            .header("User-Agent", USER_AGENT)
            .header("Openai-Intent", "conversation-edits")
            .header("x-initiator", "user")
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

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

#[cfg(test)]
mod tests {
    use super::{
        GitHubCopilotClient,
        auth::{derive_copilot_api_base_url, parse_token_expiry},
    };
    use crate::ai::{ChatRequest, Message, ThinkingConfig, ThinkingEffort, ThinkingOutput};

    #[test]
    fn parses_copilot_proxy_base_url() {
        let token = "tid=abc; proxy-ep=proxy.individual.githubcopilot.com; foo=bar";
        assert_eq!(
            derive_copilot_api_base_url(token).as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
    }

    #[test]
    fn parses_token_expiry_seconds_or_millis() {
        let seconds = serde_json::json!(1_800_000_000u64);
        let millis = serde_json::json!(1_800_000_000_000u64);
        assert_eq!(parse_token_expiry(&seconds).unwrap(), 1_800_000_000_000u64);
        assert_eq!(parse_token_expiry(&millis).unwrap(), 1_800_000_000_000u64);
    }

    #[test]
    fn converts_assistant_reasoning_fields() {
        let request = ChatRequest {
            model: "gpt-5.2-codex".to_string(),
            messages: vec![Message {
                role: "assistant".to_string(),
                content: "done".to_string(),
                thinking: Some(ThinkingOutput {
                    text: Some("reason".to_string()),
                    signature: Some("opaque".to_string()),
                    redacted: None,
                }),
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_name: None,
                tool_result: None,
                tool_error: None,
            }],
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: Some(256),
            temperature: None,
            system: None,
            thinking: Some(ThinkingConfig::enabled_with_effort(ThinkingEffort::Medium)),
        };

        let converted = GitHubCopilotClient::convert_request(request, false);
        assert_eq!(
            converted.messages[0].reasoning_text.as_deref(),
            Some("reason")
        );
        assert_eq!(
            converted.messages[0].reasoning_opaque.as_deref(),
            Some("opaque")
        );
        assert_eq!(converted.reasoning_effort, Some("medium"));
    }
}
