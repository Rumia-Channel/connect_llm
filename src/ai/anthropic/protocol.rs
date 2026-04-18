use crate::ai::{AiError, AiProvider, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingRequest>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicRequestMessage {
    pub role: String,
    pub content: Vec<AnthropicRequestContentBlock>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicRequestContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AnthropicImageSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub choice_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicThinkingRequest {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicResponse {
    pub id: String,
    pub content: Vec<AnthropicContent>,
    pub model: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub tool_use_id: Option<String>,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub is_error: Option<bool>,
}

impl AnthropicContent {
    pub(super) fn into_tool_call(self) -> Option<ToolCall> {
        if self.content_type != "tool_use" {
            return None;
        }
        Some(ToolCall {
            id: self.id?,
            name: self.name?,
            arguments: self.input.unwrap_or(Value::Null),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicStreamResponse {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<AnthropicDelta>,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block: Option<AnthropicContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<AnthropicStreamMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub partial_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<AnthropicContent>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
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

pub(super) fn api_error_from_response(status: reqwest::StatusCode, body: &str) -> AiError {
    if let Ok(error) = serde_json::from_str::<AnthropicError>(body) {
        return AiError::api(error.error.message)
            .with_provider(AiProvider::Anthropic)
            .with_status_code(status)
            .with_code(error.error.error_type)
            .with_target("/v1/messages");
    }

    AiError::api(body.to_string())
        .with_provider(AiProvider::Anthropic)
        .with_status_code(status)
        .with_target("/v1/messages")
}
