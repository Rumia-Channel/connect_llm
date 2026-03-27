#![allow(dead_code)]

pub mod anthropic;
pub mod gemini;
pub mod github_copilot;
pub mod openai;
pub mod openai_codex;
pub mod providers;

use futures_util::stream::BoxStream;
pub use github_copilot::{
    GitHubCopilotDeviceAuth, GitHubCopilotDeviceAuthOptions, github_copilot_auth_path,
    login_github_copilot_via_device,
};
pub use openai_codex::{
    OpenAiCodexBrowserAuth, OpenAiCodexBrowserAuthOptions, login_openai_codex_via_browser,
    openai_codex_auth_path,
};
use providers::{ApiStyle, ProviderSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

static DEBUG_LOGGING: AtomicBool = AtomicBool::new(false);

pub fn set_debug_logging(enabled: bool) {
    DEBUG_LOGGING.store(enabled, Ordering::Relaxed);
}

pub fn debug_logging_enabled() -> bool {
    DEBUG_LOGGING.load(Ordering::Relaxed)
}

pub(crate) fn debug_log(label: &str, body: &str) {
    if !debug_logging_enabled() {
        return;
    }

    eprintln!("[conect_llm debug] {}", label);
    eprintln!("{}", body);
}

pub(crate) fn capture_debug_json<T: Serialize>(label: &str, value: &T) -> Option<String> {
    if !debug_logging_enabled() {
        return None;
    }

    let body = serde_json::to_string_pretty(value).ok()?;
    debug_log(label, &body);
    Some(body)
}

pub(crate) fn capture_debug_text(label: &str, body: impl Into<String>) -> Option<String> {
    if !debug_logging_enabled() {
        return None;
    }

    let body = body.into();
    debug_log(label, &body);
    Some(body)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_error: Option<bool>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            tool_result: None,
            tool_error: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            tool_result: None,
            tool_error: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: String::new(),
            thinking: None,
            tool_calls,
            tool_call_id: None,
            tool_name: None,
            tool_result: None,
            tool_error: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_result: Value,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: String::new(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            tool_name: Some(tool_name.into()),
            tool_result: Some(tool_result),
            tool_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub thinking_delta: Option<String>,
    pub thinking_signature: Option<String>,
    pub tool_call_deltas: Vec<ToolCallDelta>,
    pub done: bool,
    pub debug: Option<DebugTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub content: String,
    pub model: String,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugTrace {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "empty_object")]
    pub input_schema: Value,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<Option<String>>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

impl ToolChoice {
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool(name.into())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<String>,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

pub(crate) fn parse_tool_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

pub(crate) fn serialize_tool_arguments(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => value.clone(),
        _ => serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string()),
    }
}

pub(crate) fn message_tool_result_value(message: &Message) -> Value {
    message
        .tool_result
        .clone()
        .unwrap_or_else(|| Value::String(message.content.clone()))
}

impl ThinkingOutput {
    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.signature.is_none() && self.redacted.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ThinkingEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<ThinkingDisplay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_history: Option<bool>,
}

impl ThinkingConfig {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            effort: None,
            budget_tokens: Some(1024),
            display: None,
            clear_history: None,
        }
    }

    pub fn enabled_with_effort(effort: ThinkingEffort) -> Self {
        Self {
            enabled: true,
            effort: Some(effort),
            budget_tokens: Some(1024),
            display: None,
            clear_history: None,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            effort: None,
            budget_tokens: None,
            display: None,
            clear_history: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
}

#[derive(Debug, Clone)]
pub struct AiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[async_trait::async_trait]
pub trait AiClient: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError>;
    fn chat_stream(&self, request: ChatRequest)
    -> BoxStream<'static, Result<StreamChunk, AiError>>;
    fn config(&self) -> &AiConfig;
    async fn list_models(&self) -> Result<Vec<String>, AiError>;
}

#[derive(Debug)]
pub enum AiError {
    Http(String),
    Parse(String),
    Api(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::Http(msg) => write!(f, "HTTP error: {}", msg),
            AiError::Parse(msg) => write!(f, "Parse error: {}", msg),
            AiError::Api(msg) => write!(f, "API error: {}", msg),
        }
    }
}

impl std::error::Error for AiError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiProvider {
    Anthropic,
    GitHubCopilot,
    GoogleAiStudio,
    Gemini,
    OpenAi,
    OpenAiCodex,
    Sakura,
    Kimi,
    KimiCoding,
    ZAi,
    ZAiCoding,
}

impl AiProvider {
    fn spec(&self) -> ProviderSpec {
        match self {
            AiProvider::Anthropic => providers::anthropic::spec(),
            AiProvider::GitHubCopilot => providers::github_copilot::spec(),
            AiProvider::GoogleAiStudio => providers::google_ai_studio::spec(),
            AiProvider::Gemini => providers::gemini::spec(),
            AiProvider::OpenAi => providers::openai::spec(),
            AiProvider::OpenAiCodex => providers::openai_codex::spec(),
            AiProvider::Sakura => providers::sakura::spec(),
            AiProvider::Kimi => providers::kimi::spec(),
            AiProvider::KimiCoding => providers::kimi_coding::spec(),
            AiProvider::ZAi => providers::zai::spec(),
            AiProvider::ZAiCoding => providers::zai_coding::spec(),
        }
    }

    pub fn from_index(index: i32) -> Self {
        match index {
            0 => AiProvider::Sakura,
            1 => AiProvider::Anthropic,
            2 => AiProvider::GitHubCopilot,
            3 => AiProvider::OpenAi,
            4 => AiProvider::OpenAiCodex,
            5 => AiProvider::Kimi,
            6 => AiProvider::KimiCoding,
            7 => AiProvider::ZAi,
            8 => AiProvider::ZAiCoding,
            9 => AiProvider::GoogleAiStudio,
            10 => AiProvider::Gemini,
            _ => AiProvider::Sakura,
        }
    }

    pub fn index(&self) -> i32 {
        match self {
            AiProvider::Sakura => 0,
            AiProvider::Anthropic => 1,
            AiProvider::GitHubCopilot => 2,
            AiProvider::OpenAi => 3,
            AiProvider::OpenAiCodex => 4,
            AiProvider::Kimi => 5,
            AiProvider::KimiCoding => 6,
            AiProvider::ZAi => 7,
            AiProvider::ZAiCoding => 8,
            AiProvider::GoogleAiStudio => 9,
            AiProvider::Gemini => 10,
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "Anthropic" => AiProvider::Anthropic,
            "GitHubCopilot" | "GitHub Copilot" | "Copilot" => AiProvider::GitHubCopilot,
            "GoogleAiStudio" | "Google AI Studio" => AiProvider::GoogleAiStudio,
            "Gemini" => AiProvider::Gemini,
            "OpenAi" | "OpenAI" => AiProvider::OpenAi,
            "OpenAiCodex" | "OpenAI Codex" | "Codex" => AiProvider::OpenAiCodex,
            "Sakura" => AiProvider::Sakura,
            "Kimi" => AiProvider::Kimi,
            "KimiCoding" | "Kimi Coding" => AiProvider::KimiCoding,
            "ZAi" | "Z AI" => AiProvider::ZAi,
            "ZAiCoding" | "Z AI Coding" => AiProvider::ZAiCoding,
            _ => AiProvider::Sakura,
        }
    }

    pub fn name(&self) -> &'static str {
        self.spec().name
    }

    pub fn default_base_url(&self) -> &'static str {
        self.spec().default_base_url
    }

    pub fn create_client(&self, config: AiConfig) -> Arc<dyn AiClient> {
        match self.spec().api_style {
            ApiStyle::Anthropic => {
                Arc::new(anthropic::AnthropicClient::new(config)) as Arc<dyn AiClient>
            }
            ApiStyle::Gemini => Arc::new(gemini::GeminiClient::new(config)) as Arc<dyn AiClient>,
            ApiStyle::GitHubCopilot => {
                Arc::new(github_copilot::GitHubCopilotClient::new(config)) as Arc<dyn AiClient>
            }
            ApiStyle::OpenAi => Arc::new(openai::OpenAiClient::new(config)) as Arc<dyn AiClient>,
            ApiStyle::OpenAiCodex => {
                Arc::new(openai_codex::OpenAiCodexClient::new(config)) as Arc<dyn AiClient>
            }
        }
    }

    pub fn default_model(&self) -> &'static str {
        self.spec().default_model
    }

    pub fn supports_thinking_output(&self) -> bool {
        self.spec().supports_thinking_output
    }

    pub fn supports_thinking_config(&self) -> bool {
        self.spec().supports_thinking_config
    }

    pub fn supports_tools(&self) -> bool {
        self.spec().supports_tools
    }
}
