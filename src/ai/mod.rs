#![allow(dead_code)]

pub mod anthropic;
mod auth_common;
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
use reqwest::header::HeaderMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
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

    eprintln!("[connect_llm debug] {}", label);
    eprintln!("{}", body);
}

pub(crate) fn capture_debug_json<T: Serialize>(label: &str, value: &T) -> Option<String> {
    if !debug_logging_enabled() {
        return None;
    }

    let body = serde_json::to_string_pretty(value).ok()?;
    Some(format!(
        "[connect_llm debug] {}
{}",
        label, body
    ))
}

pub(crate) fn capture_debug_text(label: &str, body: impl Into<String>) -> Option<String> {
    if !debug_logging_enabled() {
        return None;
    }

    let body = body.into();
    Some(format!(
        "[connect_llm debug] {}
{}",
        label, body
    ))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User {
        content: String,
        created_at_ms: Option<u64>,
    },
    Assistant {
        content: String,
        created_at_ms: Option<u64>,
        thinking: Option<ThinkingOutput>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        tool_name: String,
        result: Value,
        is_error: bool,
        created_at_ms: Option<u64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessageWire {
    role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    created_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(
        default,
        rename = "tool_result",
        alias = "result",
        skip_serializing_if = "Option::is_none"
    )]
    result: Option<Value>,
    #[serde(
        default,
        rename = "tool_error",
        alias = "is_error",
        skip_serializing_if = "Option::is_none"
    )]
    is_error: Option<bool>,
}

impl Message {
    fn now_timestamp_ms() -> Option<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|duration| u64::try_from(duration.as_millis()).ok())
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            created_at_ms: Self::now_timestamp_ms(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            created_at_ms: Self::now_timestamp_ms(),
            thinking: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content: String::new(),
            created_at_ms: Self::now_timestamp_ms(),
            thinking: None,
            tool_calls,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: Value,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            result,
            is_error: false,
            created_at_ms: Self::now_timestamp_ms(),
        }
    }

    pub fn with_created_at_ms(mut self, created_at_ms: u64) -> Self {
        match &mut self {
            Message::User {
                created_at_ms: slot,
                ..
            }
            | Message::Assistant {
                created_at_ms: slot,
                ..
            }
            | Message::Tool {
                created_at_ms: slot,
                ..
            } => *slot = Some(created_at_ms),
        }
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingOutput) -> Self {
        if let Message::Assistant {
            thinking: current, ..
        } = &mut self
        {
            *current = Some(thinking);
        }
        self
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        if let Message::Assistant {
            tool_calls: current,
            ..
        } = &mut self
        {
            *current = tool_calls;
        }
        self
    }

    pub fn role(&self) -> &'static str {
        match self {
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
    }

    pub fn created_at_ms(&self) -> Option<u64> {
        match self {
            Message::User { created_at_ms, .. }
            | Message::Assistant { created_at_ms, .. }
            | Message::Tool { created_at_ms, .. } => *created_at_ms,
        }
    }

    pub fn content(&self) -> Option<&str> {
        match self {
            Message::User { content, .. } | Message::Assistant { content, .. } => Some(content),
            Message::Tool { .. } => None,
        }
    }

    pub fn content_or_default(&self) -> &str {
        self.content().unwrap_or_default()
    }

    pub fn content_mut(&mut self) -> Option<&mut String> {
        match self {
            Message::User { content, .. } | Message::Assistant { content, .. } => Some(content),
            Message::Tool { .. } => None,
        }
    }

    pub fn thinking(&self) -> Option<&ThinkingOutput> {
        match self {
            Message::Assistant { thinking, .. } => thinking.as_ref(),
            Message::User { .. } | Message::Tool { .. } => None,
        }
    }

    pub fn thinking_mut(&mut self) -> Option<&mut Option<ThinkingOutput>> {
        match self {
            Message::Assistant { thinking, .. } => Some(thinking),
            Message::User { .. } | Message::Tool { .. } => None,
        }
    }

    pub fn clear_thinking(&mut self) {
        if let Some(thinking) = self.thinking_mut() {
            *thinking = None;
        }
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Message::Assistant { tool_calls, .. } => tool_calls,
            Message::User { .. } | Message::Tool { .. } => &[],
        }
    }

    pub fn tool_calls_mut(&mut self) -> Option<&mut Vec<ToolCall>> {
        match self {
            Message::Assistant { tool_calls, .. } => Some(tool_calls),
            Message::User { .. } | Message::Tool { .. } => None,
        }
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Message::Tool { tool_call_id, .. } => Some(tool_call_id),
            Message::User { .. } | Message::Assistant { .. } => None,
        }
    }

    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Message::Tool { tool_name, .. } => Some(tool_name),
            Message::User { .. } | Message::Assistant { .. } => None,
        }
    }

    pub fn tool_result_value(&self) -> Option<&Value> {
        match self {
            Message::Tool { result, .. } => Some(result),
            Message::User { .. } | Message::Assistant { .. } => None,
        }
    }

    pub fn tool_result_value_mut(&mut self) -> Option<&mut Value> {
        match self {
            Message::Tool { result, .. } => Some(result),
            Message::User { .. } | Message::Assistant { .. } => None,
        }
    }

    pub fn is_tool_error(&self) -> bool {
        match self {
            Message::Tool { is_error, .. } => *is_error,
            Message::User { .. } | Message::Assistant { .. } => false,
        }
    }

    pub fn set_tool_error(&mut self, is_error: bool) {
        if let Message::Tool {
            is_error: current, ..
        } = self
        {
            *current = is_error;
        }
    }

    pub fn as_user(&self) -> Option<&str> {
        match self {
            Message::User { content, .. } => Some(content),
            Message::Assistant { .. } | Message::Tool { .. } => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&str> {
        match self {
            Message::Assistant { content, .. } => Some(content),
            Message::User { .. } | Message::Tool { .. } => None,
        }
    }

    pub fn as_tool(&self) -> Option<(&str, &str, &Value, bool)> {
        match self {
            Message::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                ..
            } => Some((tool_call_id, tool_name, result, *is_error)),
            Message::User { .. } | Message::Assistant { .. } => None,
        }
    }
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MessageWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MessageWire::deserialize(deserializer)?;
        match wire.role.as_str() {
            "user" => Ok(Message::User {
                content: wire.content.unwrap_or_default(),
                created_at_ms: wire.created_at_ms,
            }),
            "assistant" => Ok(Message::Assistant {
                content: wire.content.unwrap_or_default(),
                created_at_ms: wire.created_at_ms,
                thinking: wire.thinking,
                tool_calls: wire.tool_calls,
            }),
            "tool" => Ok(Message::Tool {
                tool_call_id: wire.tool_call_id.unwrap_or_default(),
                tool_name: wire.tool_name.unwrap_or_else(|| "tool".to_string()),
                result: wire
                    .result
                    .or_else(|| wire.content.map(Value::String))
                    .unwrap_or(Value::Null),
                is_error: wire.is_error.unwrap_or(false),
                created_at_ms: wire.created_at_ms,
            }),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["user", "assistant", "tool"],
            )),
        }
    }
}

impl From<&Message> for MessageWire {
    fn from(message: &Message) -> Self {
        match message {
            Message::User {
                content,
                created_at_ms,
            } => Self {
                role: "user".to_string(),
                content: Some(content.clone()),
                created_at_ms: *created_at_ms,
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_name: None,
                result: None,
                is_error: None,
            },
            Message::Assistant {
                content,
                created_at_ms,
                thinking,
                tool_calls,
            } => Self {
                role: "assistant".to_string(),
                content: Some(content.clone()),
                created_at_ms: *created_at_ms,
                thinking: thinking.clone(),
                tool_calls: tool_calls.clone(),
                tool_call_id: None,
                tool_name: None,
                result: None,
                is_error: None,
            },
            Message::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                created_at_ms,
            } => Self {
                role: "tool".to_string(),
                content: Some(String::new()),
                created_at_ms: *created_at_ms,
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id.clone()),
                tool_name: Some(tool_name.clone()),
                result: Some(result.clone()),
                is_error: (*is_error).then_some(true),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub thinking_delta: Option<String>,
    pub thinking_signature: Option<String>,
    pub images: Vec<GeneratedImage>,
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
    pub images: Vec<GeneratedImage>,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratedImage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

impl GeneratedImage {
    pub fn is_empty(&self) -> bool {
        self.mime_type.is_none()
            && self.data_base64.is_none()
            && self.url.is_none()
            && self.revised_prompt.is_none()
    }

    pub fn dedup_key(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.mime_type.as_deref().unwrap_or_default(),
            self.data_base64.as_deref().unwrap_or_default(),
            self.url.as_deref().unwrap_or_default(),
            self.revised_prompt.as_deref().unwrap_or_default(),
        )
    }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
        .tool_result_value()
        .cloned()
        .unwrap_or_else(|| Value::String(message.content().unwrap_or_default().to_string()))
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
            budget_tokens: None,
            display: None,
            clear_history: None,
        }
    }

    pub fn enabled_with_effort(effort: ThinkingEffort) -> Self {
        Self {
            enabled: true,
            effort: Some(effort),
            budget_tokens: None,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiProvider {
    Anthropic,
    GitHubCopilot,
    GoogleAiStudio,
    Gemini,
    Grok,
    OpenAi,
    OpenAiCodex,
    Sakura,
    Kimi,
    KimiCoding,
    ZAi,
    ZAiCoding,
}

#[derive(Debug, Clone, Default)]
pub enum AiAuth {
    #[default]
    None,
    BearerToken(String),
    ApiKey(String),
}

#[derive(Debug, Clone)]
pub struct AiEndpointConfig {
    pub base_url: String,
}

impl AiEndpointConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AiHttpConfig {
    pub timeout: Option<Duration>,
    pub extra_headers: HeaderMap,
    pub client: Option<reqwest::Client>,
}

impl AiHttpConfig {
    pub fn build_client(&self, provider: AiProvider) -> Result<reqwest::Client, AiError> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }

        let mut builder = reqwest::Client::builder();
        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }
        if !self.extra_headers.is_empty() {
            builder = builder.default_headers(self.extra_headers.clone());
        }

        builder.build().map_err(|error| {
            AiError::configuration("failed to build HTTP client")
                .with_provider(provider)
                .with_operation("create_client")
                .with_target("reqwest::Client")
                .with_context(error.to_string())
        })
    }
}

#[derive(Debug, Clone)]
pub struct AiConfig {
    pub provider: AiProvider,
    pub endpoint: AiEndpointConfig,
    pub auth: AiAuth,
    pub default_model: String,
    pub http: AiHttpConfig,
}

impl AiConfig {
    pub fn new(provider: AiProvider) -> Self {
        Self {
            provider,
            endpoint: AiEndpointConfig::new(provider.default_base_url()),
            auth: AiAuth::None,
            default_model: provider.default_model().to_string(),
            http: AiHttpConfig::default(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.endpoint.base_url = base_url.into();
        self
    }

    pub fn with_auth(mut self, auth: AiAuth) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_http(mut self, http: AiHttpConfig) -> Self {
        self.http = http;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http.timeout = Some(timeout);
        self
    }

    pub fn with_extra_headers(mut self, extra_headers: HeaderMap) -> Self {
        self.http.extra_headers = extra_headers;
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http.client = Some(client);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.endpoint.base_url
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    pub fn bearer_token(&self) -> Option<&str> {
        match &self.auth {
            AiAuth::BearerToken(token) => Some(token.as_str()),
            AiAuth::None | AiAuth::ApiKey(_) => None,
        }
    }

    pub fn api_key(&self) -> Option<&str> {
        match &self.auth {
            AiAuth::ApiKey(token) => Some(token.as_str()),
            AiAuth::None | AiAuth::BearerToken(_) => None,
        }
    }

    pub fn require_bearer_token(&self, operation: &str) -> Result<&str, AiError> {
        match &self.auth {
            AiAuth::BearerToken(token) => Ok(token.as_str()),
            AiAuth::None => Err(AiError::auth("missing bearer token authentication")
                .with_provider(self.provider)
                .with_operation(operation)
                .with_target(self.base_url())),
            AiAuth::ApiKey(_) => Err(AiError::configuration(
                "bearer token authentication is required",
            )
            .with_provider(self.provider)
            .with_operation(operation)
            .with_target(self.base_url())),
        }
    }

    pub fn require_api_key(&self, operation: &str) -> Result<&str, AiError> {
        match &self.auth {
            AiAuth::ApiKey(token) => Ok(token.as_str()),
            AiAuth::None => Err(AiError::auth("missing API key authentication")
                .with_provider(self.provider)
                .with_operation(operation)
                .with_target(self.base_url())),
            AiAuth::BearerToken(_) => {
                Err(AiError::configuration("API key authentication is required")
                    .with_provider(self.provider)
                    .with_operation(operation)
                    .with_target(self.base_url()))
            }
        }
    }

    pub fn http_client(&self) -> Result<reqwest::Client, AiError> {
        self.http.build_client(self.provider)
    }
}

#[async_trait::async_trait]
pub trait AiClient: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError>;
    fn chat_stream(&self, request: ChatRequest)
    -> BoxStream<'static, Result<StreamChunk, AiError>>;
    fn config(&self) -> &AiConfig;
    async fn list_models(&self) -> Result<Vec<String>, AiError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiErrorKind {
    Http,
    Parse,
    Api,
    Auth,
    Configuration,
    Io,
}

impl AiErrorKind {
    fn label(&self) -> &'static str {
        match self {
            AiErrorKind::Http => "HTTP",
            AiErrorKind::Parse => "Parse",
            AiErrorKind::Api => "API",
            AiErrorKind::Auth => "Auth",
            AiErrorKind::Configuration => "Configuration",
            AiErrorKind::Io => "I/O",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AiError {
    pub kind: AiErrorKind,
    pub message: String,
    pub provider: Option<AiProvider>,
    pub operation: Option<String>,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub target: Option<String>,
}

impl AiError {
    pub fn new(kind: AiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider: None,
            operation: None,
            status: None,
            code: None,
            target: None,
        }
    }

    pub fn http(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Http, message)
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Parse, message)
    }

    pub fn api(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Api, message)
    }

    pub fn auth(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Auth, message)
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Configuration, message)
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::new(AiErrorKind::Io, message)
    }

    pub fn with_provider(mut self, provider: AiProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_status_code(mut self, status: reqwest::StatusCode) -> Self {
        self.status = Some(status.as_u16());
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    pub fn with_context(mut self, context: impl AsRef<str>) -> Self {
        let context = context.as_ref();
        if context.is_empty() {
            return self;
        }
        if self.message.is_empty() {
            self.message = context.to_string();
        } else {
            self.message = format!("{} ({})", self.message, context);
        }
        self
    }
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut scope = Vec::new();
        if let Some(provider) = self.provider {
            scope.push(provider.name().to_string());
        }
        if let Some(operation) = &self.operation {
            scope.push(operation.clone());
        }

        let mut qualifiers = Vec::new();
        if let Some(status) = self.status {
            qualifiers.push(format!("status {}", status));
        }
        if let Some(code) = &self.code {
            qualifiers.push(format!("code {}", code));
        }
        if let Some(target) = &self.target {
            qualifiers.push(format!("target {}", target));
        }

        if !scope.is_empty() {
            write!(f, "{} {} error", scope.join(" "), self.kind.label())?;
        } else {
            write!(f, "{} error", self.kind.label())?;
        }

        if !qualifiers.is_empty() {
            write!(f, " [{}]", qualifiers.join(", "))?;
        }

        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }

        Ok(())
    }
}

impl std::error::Error for AiError {}

impl AiProvider {
    fn spec(&self) -> ProviderSpec {
        match self {
            AiProvider::Anthropic => providers::anthropic::spec(),
            AiProvider::GitHubCopilot => providers::github_copilot::spec(),
            AiProvider::GoogleAiStudio => providers::google_ai_studio::spec(),
            AiProvider::Gemini => providers::gemini::spec(),
            AiProvider::Grok => providers::grok::spec(),
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
            3 => AiProvider::Grok,
            4 => AiProvider::OpenAi,
            5 => AiProvider::OpenAiCodex,
            6 => AiProvider::Kimi,
            7 => AiProvider::KimiCoding,
            8 => AiProvider::ZAi,
            9 => AiProvider::ZAiCoding,
            10 => AiProvider::GoogleAiStudio,
            11 => AiProvider::Gemini,
            _ => AiProvider::Sakura,
        }
    }

    pub fn index(&self) -> i32 {
        match self {
            AiProvider::Sakura => 0,
            AiProvider::Anthropic => 1,
            AiProvider::GitHubCopilot => 2,
            AiProvider::Grok => 3,
            AiProvider::OpenAi => 4,
            AiProvider::OpenAiCodex => 5,
            AiProvider::Kimi => 6,
            AiProvider::KimiCoding => 7,
            AiProvider::ZAi => 8,
            AiProvider::ZAiCoding => 9,
            AiProvider::GoogleAiStudio => 10,
            AiProvider::Gemini => 11,
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "Anthropic" => AiProvider::Anthropic,
            "GitHubCopilot" | "GitHub Copilot" | "Copilot" => AiProvider::GitHubCopilot,
            "GoogleAiStudio" | "Google AI Studio" => AiProvider::GoogleAiStudio,
            "Gemini" => AiProvider::Gemini,
            "Grok" | "xAI" | "XAI" | "X Grok" => AiProvider::Grok,
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

    pub fn create_client(&self, mut config: AiConfig) -> Result<Arc<dyn AiClient>, AiError> {
        config.provider = *self;
        if config.endpoint.base_url.trim().is_empty() {
            config.endpoint.base_url = self.default_base_url().to_string();
        }
        if config.default_model.trim().is_empty() {
            config.default_model = self.default_model().to_string();
        }

        match self.spec().api_style {
            ApiStyle::Anthropic => {
                Ok(Arc::new(anthropic::AnthropicClient::new(config)?) as Arc<dyn AiClient>)
            }
            ApiStyle::Gemini => {
                Ok(Arc::new(gemini::GeminiClient::new(config)?) as Arc<dyn AiClient>)
            }
            ApiStyle::GitHubCopilot => {
                Ok(Arc::new(github_copilot::GitHubCopilotClient::new(config)?)
                    as Arc<dyn AiClient>)
            }
            ApiStyle::OpenAi => {
                Ok(Arc::new(openai::OpenAiClient::new(config)?) as Arc<dyn AiClient>)
            }
            ApiStyle::OpenAiCodex => {
                Ok(Arc::new(openai_codex::OpenAiCodexClient::new(config)?) as Arc<dyn AiClient>)
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

#[cfg(test)]
mod tests {
    use super::Message;
    use serde_json::json;

    #[test]
    fn serializes_tool_messages_with_legacy_wire_fields() {
        let mut message = Message::tool_result("call_1", "lookup", json!({"ok": true}));
        message.set_tool_error(true);

        let value = serde_json::to_value(&message).expect("serialize tool message");

        assert_eq!(value.get("role"), Some(&json!("tool")));
        assert_eq!(value.get("content"), Some(&json!("")));
        assert_eq!(value.get("tool_result"), Some(&json!({"ok": true})));
        assert_eq!(value.get("tool_error"), Some(&json!(true)));
        assert!(value.get("result").is_none());
        assert!(value.get("is_error").is_none());
    }

    #[test]
    fn deserializes_legacy_tool_message_shape() {
        let value = json!({
            "role": "tool",
            "content": "",
            "tool_call_id": "call_1",
            "tool_name": "lookup",
            "tool_result": { "ok": true },
            "tool_error": true
        });

        let message: Message =
            serde_json::from_value(value).expect("deserialize legacy tool message");
        let (tool_call_id, tool_name, result, is_error) = message.as_tool().expect("tool message");

        assert_eq!(tool_call_id, "call_1");
        assert_eq!(tool_name, "lookup");
        assert_eq!(result, &json!({"ok": true}));
        assert!(is_error);
    }
}
