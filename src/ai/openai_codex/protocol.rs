use crate::ai::{AiError, AiProvider, ToolCall, parse_tool_arguments};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexRequest {
    pub model: String,
    pub instructions: String,
    pub input: Vec<OpenAiCodexInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenAiCodexReasoningRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiCodexTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAiCodexToolChoice>,
    pub stream: bool,
    pub store: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexReasoningRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum OpenAiCodexInputItem {
    Message(OpenAiCodexInputMessage),
    FunctionCall(OpenAiCodexFunctionCallItem),
    FunctionCallOutput(OpenAiCodexFunctionCallOutputItem),
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexInputMessage {
    pub role: String,
    pub content: Vec<OpenAiCodexInputContent>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexFunctionCallItem {
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexFunctionCallOutputItem {
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub call_id: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpenAiCodexTool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum OpenAiCodexToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexResponse {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub usage: Option<OpenAiCodexUsage>,
    #[serde(default)]
    pub output: Vec<OpenAiCodexOutputItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexInputContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexOutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub content: Vec<OpenAiCodexOutputContent>,
    #[serde(default)]
    pub summary: Vec<OpenAiCodexReasoningSummary>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct OpenAiCodexUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexOutputContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexReasoningSummary {
    #[serde(rename = "type", default)]
    pub summary_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct OpenAiCodexEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub output_index: Option<usize>,
    #[serde(default)]
    pub item_id: Option<String>,
    #[serde(default)]
    pub summary_index: Option<usize>,
    #[serde(default)]
    pub delta: Option<String>,
    #[serde(default)]
    pub item: Option<OpenAiCodexOutputItem>,
    #[serde(default)]
    pub response: Option<OpenAiCodexResponse>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct PendingToolCallState {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
    pub saw_argument_delta: bool,
}

impl PendingToolCallState {
    pub(super) fn into_tool_call(self, index: usize) -> Option<ToolCall> {
        Some(ToolCall {
            id: self.id.unwrap_or_else(|| format!("tool-call-{}", index)),
            name: self.name?,
            arguments: parse_tool_arguments(&self.arguments),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexError {
    error: OpenAiCodexErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexErrorDetail {
    message: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
}

pub(super) fn api_error_from_response(status: reqwest::StatusCode, body: &str) -> AiError {
    if let Ok(error) = serde_json::from_str::<OpenAiCodexError>(body) {
        let mut structured = AiError::api(error.error.message.clone())
            .with_provider(AiProvider::OpenAiCodex)
            .with_status_code(status)
            .with_target("/codex/responses");
        if let Some(code) = error.error.code.clone() {
            structured = structured.with_code(code);
        }
        if let Some(error_type) = error.error.error_type.clone() {
            structured = structured.with_context(error_type);
        }
        return structured;
    }

    AiError::api(body.to_string())
        .with_provider(AiProvider::OpenAiCodex)
        .with_status_code(status)
        .with_target("/codex/responses")
}
