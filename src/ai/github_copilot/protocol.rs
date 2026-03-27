use crate::ai::{AiError, ToolCall, ToolCallDelta, parse_tool_arguments};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub(super) struct GitHubCopilotRequest {
    pub model: String,
    pub messages: Vec<GitHubCopilotMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GitHubCopilotToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct GitHubCopilotMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_opaque: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<GitHubCopilotToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct GitHubCopilotToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: GitHubCopilotFunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct GitHubCopilotFunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: GitHubCopilotToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<GitHubCopilotChoice>,
    #[serde(default)]
    pub usage: Option<GitHubCopilotUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotChoice {
    pub message: GitHubCopilotMessageResponse,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotMessageResponse {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_text: Option<String>,
    #[serde(default)]
    pub reasoning_opaque: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<GitHubCopilotToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct GitHubCopilotUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotStreamResponse {
    #[allow(dead_code)]
    pub id: Option<String>,
    pub choices: Vec<GitHubCopilotStreamChoice>,
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotStreamChoice {
    pub delta: GitHubCopilotDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct GitHubCopilotDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_text: Option<String>,
    #[serde(default)]
    pub reasoning_opaque: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<GitHubCopilotToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<GitHubCopilotToolFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GitHubCopilotToolFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotErrorEnvelope {
    error: GitHubCopilotErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotErrorDetail {
    message: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
}

fn format_error_detail(status: reqwest::StatusCode, detail: &GitHubCopilotErrorDetail) -> String {
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

pub(super) fn api_error_from_response(status: reqwest::StatusCode, body: &str) -> AiError {
    if let Ok(error) = serde_json::from_str::<GitHubCopilotErrorEnvelope>(body) {
        return AiError::Api(format_error_detail(status, &error.error));
    }

    AiError::Api(format!("HTTP {}: {}", status, body))
}

pub(super) fn parse_tool_calls(tool_calls: Option<Vec<GitHubCopilotToolCall>>) -> Vec<ToolCall> {
    tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tool_call| ToolCall {
            id: tool_call.id,
            name: tool_call.function.name,
            arguments: parse_tool_arguments(&tool_call.function.arguments),
        })
        .collect()
}

pub(super) fn parse_tool_call_deltas(
    tool_calls: Option<Vec<GitHubCopilotToolCallDelta>>,
) -> Vec<ToolCallDelta> {
    tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tool_call| ToolCallDelta {
            index: tool_call.index,
            id: tool_call.id,
            name: tool_call
                .function
                .as_ref()
                .and_then(|function| function.name.clone()),
            arguments: tool_call.function.and_then(|function| function.arguments),
        })
        .collect()
}
