#![allow(dead_code)]

pub mod anthropic;
pub mod openai;

use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub content: String,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
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
    OpenAi,
    Sakura,
    Kimi,
    KimiCoding,
    ZAi,
    ZAiCoding,
}

impl AiProvider {
    pub fn from_index(index: i32) -> Self {
        match index {
            0 => AiProvider::Sakura,
            1 => AiProvider::Anthropic,
            2 => AiProvider::OpenAi,
            3 => AiProvider::Kimi,
            4 => AiProvider::KimiCoding,
            5 => AiProvider::ZAi,
            6 => AiProvider::ZAiCoding,
            _ => AiProvider::Sakura,
        }
    }

    pub fn index(&self) -> i32 {
        match self {
            AiProvider::Sakura => 0,
            AiProvider::Anthropic => 1,
            AiProvider::OpenAi => 2,
            AiProvider::Kimi => 3,
            AiProvider::KimiCoding => 4,
            AiProvider::ZAi => 5,
            AiProvider::ZAiCoding => 6,
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "Anthropic" => AiProvider::Anthropic,
            "OpenAi" | "OpenAI" => AiProvider::OpenAi,
            "Sakura" => AiProvider::Sakura,
            "Kimi" => AiProvider::Kimi,
            "KimiCoding" | "Kimi Coding" => AiProvider::KimiCoding,
            "ZAi" | "Z AI" => AiProvider::ZAi,
            "ZAiCoding" | "Z AI Coding" => AiProvider::ZAiCoding,
            _ => AiProvider::Sakura,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            AiProvider::Sakura => "Sakura",
            AiProvider::Anthropic => "Anthropic",
            AiProvider::OpenAi => "OpenAi",
            AiProvider::Kimi => "Kimi",
            AiProvider::KimiCoding => "KimiCoding",
            AiProvider::ZAi => "ZAi",
            AiProvider::ZAiCoding => "ZAiCoding",
        }
    }

    pub fn default_base_url(&self) -> &'static str {
        match self {
            AiProvider::Anthropic => "https://api.anthropic.com",
            AiProvider::OpenAi => "https://api.openai.com",
            AiProvider::Sakura => "https://api.ai.sakura.ad.jp",
            AiProvider::Kimi => "https://api.moonshot.ai",
            AiProvider::KimiCoding => "https://api.kimi.com/coding",
            AiProvider::ZAi => "https://api.z.ai/api/paas/v4",
            AiProvider::ZAiCoding => "https://api.z.ai/api/coding/paas/v4",
        }
    }

    pub fn create_client(&self, config: AiConfig) -> Arc<dyn AiClient> {
        match self {
            AiProvider::Anthropic | AiProvider::KimiCoding => {
                Arc::new(anthropic::AnthropicClient::new(config)) as Arc<dyn AiClient>
            }
            AiProvider::OpenAi
            | AiProvider::Sakura
            | AiProvider::Kimi
            | AiProvider::ZAi
            | AiProvider::ZAiCoding => {
                Arc::new(openai::OpenAiClient::new(config)) as Arc<dyn AiClient>
            }
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            AiProvider::Anthropic => "claude-3-5-sonnet-20241022",
            AiProvider::OpenAi => "gpt-4o",
            AiProvider::Sakura => "preview/Kimi-K2.5",
            AiProvider::Kimi => "kimi-k2.5",
            AiProvider::KimiCoding => "kimi-for-coding",
            AiProvider::ZAi => "glm-5",
            AiProvider::ZAiCoding => "glm-4.7",
        }
    }
}
