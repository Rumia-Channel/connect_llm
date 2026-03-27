use crate::ai::{AiError, ToolCall, ToolCallDelta, parse_tool_arguments};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<OpenAiThinkingRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<OpenAiExtraBody>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: OpenAiFunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiFunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiThinkingRequest {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiExtraBody {
    pub google: OpenAiGoogleExtraBody,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiGoogleExtraBody {
    pub thinking_config: OpenAiGoogleThinkingConfig,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiGoogleThinkingConfig {
    pub include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiResponse {
    pub id: String,
    pub choices: Vec<OpenAiChoice>,
    pub model: String,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiChoice {
    pub message: OpenAiMessageResponse,
    #[serde(default)]
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiMessageResponse {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiStreamResponse {
    #[allow(dead_code)]
    pub id: String,
    pub choices: Vec<OpenAiStreamChoice>,
    #[allow(dead_code)]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiStreamChoice {
    pub delta: OpenAiDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAiToolFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiToolFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ModelInfo {
    pub id: String,
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

pub(super) fn api_error_from_response(
    status: reqwest::StatusCode,
    body: &str,
    base_url: &str,
) -> AiError {
    if let Ok(error) = serde_json::from_str::<OpenAiError>(body) {
        return AiError::Api(format_error_detail(status, base_url, &error.error));
    }

    AiError::Api(format!("HTTP {}: {}", status, body))
}

pub(super) fn convert_tool_calls_to_response(
    tool_calls: Option<Vec<OpenAiToolCall>>,
) -> Vec<ToolCall> {
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

pub(super) fn convert_tool_call_deltas(
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
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

pub(super) fn auto_tool_choice() -> Value {
    json!("auto")
}
