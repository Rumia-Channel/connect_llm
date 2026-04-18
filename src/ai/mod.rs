#![allow(dead_code)]

pub mod anthropic;
mod auth_common;
pub mod gemini;
pub mod github_copilot;
pub mod openai;
pub mod openai_codex;
pub mod providers;

use futures_util::{StreamExt, stream::BoxStream};
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        mime_type: String,
        data_base64: String,
    },
    Url {
        url: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputImage {
    pub source: ImageSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl InputImage {
    pub fn from_base64(mime_type: impl Into<String>, data_base64: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64 {
                mime_type: mime_type.into(),
                data_base64: data_base64.into(),
            },
            detail: None,
        }
    }

    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url { url: url.into() },
            detail: None,
        }
    }

    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    pub fn as_url(&self) -> Option<&str> {
        match &self.source {
            ImageSource::Url { url } => Some(url),
            ImageSource::Base64 { .. } => None,
        }
    }

    pub fn as_data_url(&self) -> Option<String> {
        match &self.source {
            ImageSource::Base64 {
                mime_type,
                data_base64,
            } => Some(format!("data:{};base64,{}", mime_type, data_base64)),
            ImageSource::Url { url } => Some(url.clone()),
        }
    }

    pub(crate) fn as_inline_base64(&self) -> Option<(String, String)> {
        match &self.source {
            ImageSource::Base64 {
                mime_type,
                data_base64,
            } => Some((mime_type.clone(), data_base64.clone())),
            ImageSource::Url { url } => parse_data_url(url),
        }
    }

    pub fn is_url(&self) -> bool {
        matches!(self.source, ImageSource::Url { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { image: InputImage },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image(image: InputImage) -> Self {
        Self::Image { image }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } => Some(text),
            ContentPart::Image { .. } => None,
        }
    }

    pub fn as_image(&self) -> Option<&InputImage> {
        match self {
            ContentPart::Text { .. } => None,
            ContentPart::Image { image } => Some(image),
        }
    }
}

impl From<String> for ContentPart {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<&str> for ContentPart {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RequestMessage {
    User {
        content: Vec<ContentPart>,
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

impl RequestMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self::user_parts(vec![ContentPart::text(content)])
    }

    pub fn user_parts(content: Vec<ContentPart>) -> Self {
        Self::User {
            content,
            created_at_ms: Message::now_timestamp_ms(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            created_at_ms: Message::now_timestamp_ms(),
            thinking: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content: String::new(),
            created_at_ms: Message::now_timestamp_ms(),
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
            created_at_ms: Message::now_timestamp_ms(),
        }
    }

    pub fn with_created_at_ms(mut self, created_at_ms: u64) -> Self {
        match &mut self {
            RequestMessage::User {
                created_at_ms: slot,
                ..
            }
            | RequestMessage::Assistant {
                created_at_ms: slot,
                ..
            }
            | RequestMessage::Tool {
                created_at_ms: slot,
                ..
            } => *slot = Some(created_at_ms),
        }
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingOutput) -> Self {
        if let RequestMessage::Assistant {
            thinking: current, ..
        } = &mut self
        {
            *current = Some(thinking);
        }
        self
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        if let RequestMessage::Assistant {
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
            RequestMessage::User { .. } => "user",
            RequestMessage::Assistant { .. } => "assistant",
            RequestMessage::Tool { .. } => "tool",
        }
    }

    pub fn created_at_ms(&self) -> Option<u64> {
        match self {
            RequestMessage::User { created_at_ms, .. }
            | RequestMessage::Assistant { created_at_ms, .. }
            | RequestMessage::Tool { created_at_ms, .. } => *created_at_ms,
        }
    }

    pub fn content_parts(&self) -> Option<&[ContentPart]> {
        match self {
            RequestMessage::User { content, .. } => Some(content),
            RequestMessage::Assistant { .. } | RequestMessage::Tool { .. } => None,
        }
    }

    pub fn content_parts_mut(&mut self) -> Option<&mut Vec<ContentPart>> {
        match self {
            RequestMessage::User { content, .. } => Some(content),
            RequestMessage::Assistant { .. } | RequestMessage::Tool { .. } => None,
        }
    }

    pub fn content_text(&self) -> Option<String> {
        match self {
            RequestMessage::User { content, .. } => Some(join_text_parts(content)),
            RequestMessage::Assistant { content, .. } => Some(content.clone()),
            RequestMessage::Tool { .. } => None,
        }
    }

    pub fn contains_input_images(&self) -> bool {
        matches!(
            self,
            RequestMessage::User { content, .. }
                if content.iter().any(|part| matches!(part, ContentPart::Image { .. }))
        )
    }

    pub fn thinking(&self) -> Option<&ThinkingOutput> {
        match self {
            RequestMessage::Assistant { thinking, .. } => thinking.as_ref(),
            RequestMessage::User { .. } | RequestMessage::Tool { .. } => None,
        }
    }

    pub fn clear_thinking(&mut self) {
        if let RequestMessage::Assistant { thinking, .. } = self {
            *thinking = None;
        }
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            RequestMessage::Assistant { tool_calls, .. } => tool_calls,
            RequestMessage::User { .. } | RequestMessage::Tool { .. } => &[],
        }
    }
}

impl From<Message> for RequestMessage {
    fn from(message: Message) -> Self {
        match message {
            Message::User {
                content,
                created_at_ms,
            } => Self::User {
                content: vec![ContentPart::text(content)],
                created_at_ms,
            },
            Message::Assistant {
                content,
                created_at_ms,
                thinking,
                tool_calls,
            } => Self::Assistant {
                content,
                created_at_ms,
                thinking,
                tool_calls,
            },
            Message::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                created_at_ms,
            } => Self::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                created_at_ms,
            },
        }
    }
}

impl From<&Message> for RequestMessage {
    fn from(message: &Message) -> Self {
        message.clone().into()
    }
}

impl TryFrom<RequestMessage> for Message {
    type Error = AiError;

    fn try_from(message: RequestMessage) -> Result<Self, Self::Error> {
        match message {
            RequestMessage::User {
                content,
                created_at_ms,
            } => {
                if content.iter().any(|part| part.as_image().is_some()) {
                    return Err(AiError::configuration(
                        "cannot downgrade user message with input images to text-only Message",
                    ));
                }
                Ok(Message::User {
                    content: join_text_parts(&content),
                    created_at_ms,
                })
            }
            RequestMessage::Assistant {
                content,
                created_at_ms,
                thinking,
                tool_calls,
            } => Ok(Message::Assistant {
                content,
                created_at_ms,
                thinking,
                tool_calls,
            }),
            RequestMessage::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                created_at_ms,
            } => Ok(Message::Tool {
                tool_call_id,
                tool_name,
                result,
                is_error,
                created_at_ms,
            }),
        }
    }
}

impl TryFrom<&RequestMessage> for Message {
    type Error = AiError;

    fn try_from(message: &RequestMessage) -> Result<Self, Self::Error> {
        message.clone().try_into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalChatRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
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

impl MultimodalChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<RequestMessage>) -> Self {
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

    pub fn has_input_images(&self) -> bool {
        self.messages
            .iter()
            .any(RequestMessage::contains_input_images)
    }

    pub fn try_into_chat_request(self) -> Result<ChatRequest, AiError> {
        Ok(ChatRequest {
            model: self.model,
            messages: self
                .messages
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()?,
            tools: self.tools,
            tool_choice: self.tool_choice,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            system: self.system,
            thinking: self.thinking,
        })
    }
}

impl From<ChatRequest> for MultimodalChatRequest {
    fn from(request: ChatRequest) -> Self {
        Self {
            model: request.model,
            messages: request.messages.into_iter().map(Into::into).collect(),
            tools: request.tools,
            tool_choice: request.tool_choice,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            system: request.system,
            thinking: request.thinking,
        }
    }
}

impl From<&ChatRequest> for MultimodalChatRequest {
    fn from(request: &ChatRequest) -> Self {
        request.clone().into()
    }
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

fn join_text_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(ContentPart::as_text)
        .collect::<Vec<_>>()
        .join("")
}

fn parse_data_url(url: &str) -> Option<(String, String)> {
    let encoded = url.strip_prefix("data:")?;
    let (metadata, data_base64) = encoded.split_once(";base64,")?;
    let mime_type = metadata.trim();
    if mime_type.is_empty() || data_base64.is_empty() {
        return None;
    }
    Some((mime_type.to_string(), data_base64.to_string()))
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
    async fn chat_multimodal(
        &self,
        request: MultimodalChatRequest,
    ) -> Result<ChatResponse, AiError> {
        if request.has_input_images() {
            return Err(unsupported_input_images_error(
                self.config().provider,
                "chat_multimodal",
                self.config().base_url(),
            ));
        }
        self.chat(request.try_into_chat_request()?).await
    }
    fn chat_multimodal_stream(
        &self,
        request: MultimodalChatRequest,
    ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
        if request.has_input_images() {
            let provider = self.config().provider;
            let target = self.config().base_url().to_string();
            return futures_util::stream::once(async move {
                Err(unsupported_input_images_error(
                    provider,
                    "chat_multimodal_stream",
                    target,
                ))
            })
            .boxed();
        }

        match request.try_into_chat_request() {
            Ok(request) => self.chat_stream(request),
            Err(error) => futures_util::stream::once(async move { Err(error) }).boxed(),
        }
    }
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

pub(crate) fn unsupported_input_images_error(
    provider: AiProvider,
    operation: &str,
    target: impl Into<String>,
) -> AiError {
    AiError::configuration("input images are not supported by this provider")
        .with_provider(provider)
        .with_operation(operation)
        .with_target(target)
}

pub(crate) fn unsupported_image_url_error(
    provider: AiProvider,
    operation: &str,
    target: impl Into<String>,
) -> AiError {
    AiError::configuration(
        "remote image URLs are not supported by this provider; send base64 data instead",
    )
    .with_provider(provider)
    .with_operation(operation)
    .with_target(target)
}

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

    pub fn supports_input_images(&self) -> bool {
        !matches!(
            self.spec().api_style,
            ApiStyle::GitHubCopilot | ApiStyle::OpenAiCodex
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AiClient, AiConfig, AiError, AiProvider, ChatRequest, ChatResponse, ContentPart,
        ImageDetail, InputImage, Message, MultimodalChatRequest, RequestMessage, StreamChunk,
        Usage,
    };
    use futures_util::{
        StreamExt,
        stream::{self, BoxStream},
    };
    use serde_json::json;

    struct DummyClient {
        config: AiConfig,
    }

    #[async_trait::async_trait]
    impl AiClient for DummyClient {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
            Ok(ChatResponse {
                id: "dummy".to_string(),
                content: request.messages[0].content_or_default().to_string(),
                model: request.model,
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
                thinking: None,
                images: Vec::new(),
                tool_calls: Vec::new(),
                debug: None,
            })
        }

        fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
            stream::empty().boxed()
        }

        fn config(&self) -> &AiConfig {
            &self.config
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec![self.config.default_model.clone()])
        }
    }

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

    #[test]
    fn converts_text_only_request_into_multimodal_request() {
        let request = ChatRequest::new("model", vec![Message::user("hello")]);
        let multimodal = MultimodalChatRequest::from(request);

        assert_eq!(multimodal.messages.len(), 1);
        assert!(!multimodal.has_input_images());
        assert_eq!(
            multimodal.messages[0]
                .content_parts()
                .and_then(|parts| parts[0].as_text()),
            Some("hello")
        );
    }

    #[test]
    fn request_message_text_only_downgrades_back_to_message() {
        let message = RequestMessage::user_parts(vec![
            ContentPart::text("hello"),
            ContentPart::text(" world"),
        ]);

        let downgraded: Message = message.try_into().expect("downgrade to text-only message");
        assert_eq!(downgraded.as_user(), Some("hello world"));
    }

    #[test]
    fn request_message_with_images_cannot_downgrade_to_text_only() {
        let message = RequestMessage::user_parts(vec![
            ContentPart::text("hello"),
            ContentPart::image(
                InputImage::from_base64("image/png", "aGVsbG8=").with_detail(ImageDetail::Low),
            ),
        ]);

        let error = Message::try_from(message).expect_err("image-bearing message should fail");
        assert_eq!(error.kind, super::AiErrorKind::Configuration);
    }

    #[tokio::test]
    async fn unsupported_provider_rejects_multimodal_images_explicitly() {
        let client = DummyClient {
            config: AiConfig::new(AiProvider::GitHubCopilot),
        };
        let request = MultimodalChatRequest::new(
            "model",
            vec![RequestMessage::user_parts(vec![ContentPart::image(
                InputImage::from_url("https://example.com/cat.png"),
            )])],
        );

        let error = client
            .chat_multimodal(request)
            .await
            .expect_err("unsupported provider should reject input images");

        assert_eq!(error.kind, super::AiErrorKind::Configuration);
        assert_eq!(error.provider, Some(AiProvider::GitHubCopilot));
    }

    #[test]
    fn provider_capability_marks_text_only_providers() {
        assert!(AiProvider::OpenAi.supports_input_images());
        assert!(AiProvider::Anthropic.supports_input_images());
        assert!(AiProvider::Gemini.supports_input_images());
        assert!(!AiProvider::GitHubCopilot.supports_input_images());
        assert!(!AiProvider::OpenAiCodex.supports_input_images());
    }
}
