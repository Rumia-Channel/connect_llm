#![allow(dead_code)]

use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, DebugTrace, StreamChunk,
    ThinkingEffort, ThinkingOutput, ToolCall, ToolCallDelta, ToolChoice, ToolDefinition, Usage,
    capture_debug_json, capture_debug_text, parse_tool_arguments, serialize_tool_arguments,
};
use futures_util::StreamExt;
use reqwest::{Client, blocking::Client as BlockingClient};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";
const DEFAULT_DOMAIN: &str = "github.com";
const DEFAULT_AUTH_TIMEOUT_SECS: u64 = 900;
const OAUTH_POLLING_SAFETY_MARGIN_MS: u64 = 3_000;
const TOKEN_REFRESH_SAFETY_WINDOW_MS: u64 = 5 * 60 * 1_000;
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const DEFAULT_COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";
const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone)]
pub struct GitHubCopilotDeviceAuthOptions {
    pub domain: String,
    pub timeout: Duration,
    pub auth_path: Option<PathBuf>,
    pub open_browser: bool,
}

impl Default for GitHubCopilotDeviceAuthOptions {
    fn default() -> Self {
        Self {
            domain: DEFAULT_DOMAIN.to_string(),
            timeout: Duration::from_secs(DEFAULT_AUTH_TIMEOUT_SECS),
            auth_path: None,
            open_browser: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GitHubCopilotDeviceAuth {
    pub github_token: String,
    pub auth_path: PathBuf,
    pub domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GitHubCopilotAuthFile {
    #[serde(default)]
    provider: Option<String>,
    github_token: String,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    copilot_api_token: Option<String>,
    #[serde(default)]
    copilot_api_token_expires_at_ms: Option<u64>,
    #[serde(default)]
    copilot_api_base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GitHubCopilotRequest {
    model: String,
    messages: Vec<GitHubCopilotMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GitHubCopilotToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize)]
struct GitHubCopilotMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_opaque: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<GitHubCopilotToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
struct GitHubCopilotToolDefinition {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: GitHubCopilotFunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
struct GitHubCopilotFunctionDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: GitHubCopilotToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotResponse {
    id: String,
    model: String,
    choices: Vec<GitHubCopilotChoice>,
    #[serde(default)]
    usage: Option<GitHubCopilotUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotChoice {
    message: GitHubCopilotMessageResponse,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotMessageResponse {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
    #[serde(default)]
    reasoning_opaque: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<GitHubCopilotToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GitHubCopilotUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotStreamResponse {
    #[allow(dead_code)]
    id: Option<String>,
    choices: Vec<GitHubCopilotStreamChoice>,
    #[allow(dead_code)]
    model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotStreamChoice {
    delta: GitHubCopilotDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GitHubCopilotDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
    #[serde(default)]
    reasoning_opaque: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<GitHubCopilotToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<GitHubCopilotToolFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubCopilotToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubDeviceCodeResponse {
    verification_uri: String,
    user_code: String,
    device_code: String,
    interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubDeviceTokenResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    interval: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CopilotTokenExchangeResponse {
    token: String,
    expires_at: serde_json::Value,
}

#[derive(Debug, Clone)]
struct ResolvedCopilotAuth {
    api_token: String,
    base_url: String,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn home_dir() -> Result<PathBuf, AiError> {
    if let Some(path) = env::var_os("HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("USERPROFILE").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    Err(AiError::Http(
        "could not resolve HOME/USERPROFILE for GitHub Copilot auth".to_string(),
    ))
}

pub fn github_copilot_auth_path() -> Result<PathBuf, AiError> {
    if let Some(path) = env::var_os("COPILOT_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path).join("auth.json"));
    }

    Ok(home_dir()?.join(".copilot").join("auth.json"))
}

fn normalize_domain(domain: &str) -> String {
    domain
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
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

fn api_error_from_response(status: reqwest::StatusCode, body: &str) -> AiError {
    if let Ok(error) = serde_json::from_str::<GitHubCopilotErrorEnvelope>(body) {
        return AiError::Api(format_error_detail(status, &error.error));
    }

    AiError::Api(format!("HTTP {}: {}", status, body))
}

fn derive_copilot_api_base_url(token: &str) -> Option<String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }

    let proxy_ep = trimmed
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep="))
        .map(str::trim)?;
    let host = proxy_ep
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("proxy.")
        .trim();

    if host.is_empty() {
        return None;
    }

    Some(format!("https://api.{}", host))
}

fn parse_token_expiry(expires_at: &serde_json::Value) -> Result<u64, AiError> {
    match expires_at {
        serde_json::Value::Number(number) => number
            .as_u64()
            .map(|value| {
                if value < 100_000_000_000 {
                    value * 1_000
                } else {
                    value
                }
            })
            .ok_or_else(|| AiError::Parse("invalid Copilot expires_at".to_string())),
        serde_json::Value::String(value) => value
            .parse::<u64>()
            .map(|value| {
                if value < 100_000_000_000 {
                    value * 1_000
                } else {
                    value
                }
            })
            .map_err(|error| AiError::Parse(error.to_string())),
        _ => Err(AiError::Parse(
            "Copilot token response missing expires_at".to_string(),
        )),
    }
}

fn load_auth_file(path: &Path) -> Result<GitHubCopilotAuthFile, AiError> {
    let body = fs::read_to_string(path).map_err(|error| AiError::Http(error.to_string()))?;
    serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))
}

fn save_auth_file(path: &Path, auth: &GitHubCopilotAuthFile) -> Result<(), AiError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| AiError::Http(error.to_string()))?;
    }

    let body =
        serde_json::to_string_pretty(auth).map_err(|error| AiError::Parse(error.to_string()))?;
    let mut file = fs::File::create(path).map_err(|error| AiError::Http(error.to_string()))?;
    file.write_all(body.as_bytes())
        .map_err(|error| AiError::Http(error.to_string()))?;
    Ok(())
}

fn open_url_in_browser(url: &str) -> Result<(), AiError> {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(|error| {
                AiError::Api(format!(
                    "Failed to open browser automatically: {}. Open this URL manually: {}",
                    error, url
                ))
            })?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn().map_err(|error| {
            AiError::Api(format!(
                "Failed to open browser automatically: {}. Open this URL manually: {}",
                error, url
            ))
        })?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(url).spawn().map_err(|error| {
            AiError::Api(format!(
                "Failed to open browser automatically: {}. Open this URL manually: {}",
                error, url
            ))
        })?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err(AiError::Api(format!(
        "Automatic browser opening is not supported on this platform. Open this URL manually: {}",
        url
    )))
}

pub fn login_github_copilot_via_device(
    options: GitHubCopilotDeviceAuthOptions,
) -> Result<GitHubCopilotDeviceAuth, AiError> {
    let auth_path = options
        .auth_path
        .clone()
        .map(Ok)
        .unwrap_or_else(github_copilot_auth_path)?;
    let domain = normalize_domain(&options.domain);
    let device_code_url = format!("https://{}/login/device/code", domain);
    let access_token_url = format!("https://{}/login/oauth/access_token", domain);
    let client = BlockingClient::new();

    let device_response = client
        .post(&device_code_url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .header("User-Agent", USER_AGENT)
        .json(&serde_json::json!({
            "client_id": CLIENT_ID,
            "scope": "read:user",
        }))
        .send()
        .map_err(|error| AiError::Http(error.to_string()))?;

    let device_status = device_response.status();
    let device_body = device_response
        .text()
        .map_err(|error| AiError::Http(error.to_string()))?;

    if !device_status.is_success() {
        return Err(AiError::Api(format!(
            "GitHub device auth start failed: HTTP {}: {}",
            device_status, device_body
        )));
    }

    let device_data: GitHubDeviceCodeResponse =
        serde_json::from_str(&device_body).map_err(|error| AiError::Parse(error.to_string()))?;

    eprintln!("GitHub Copilot device login");
    eprintln!("Visit: {}", device_data.verification_uri);
    eprintln!("Code: {}", device_data.user_code);
    if options.open_browser {
        if let Err(error) = open_url_in_browser(&device_data.verification_uri) {
            eprintln!("{}", error);
        }
    }
    eprintln!("Waiting for authorization...");

    let deadline = std::time::Instant::now() + options.timeout;

    loop {
        if std::time::Instant::now() >= deadline {
            return Err(AiError::Http(
                "GitHub Copilot device login timed out".to_string(),
            ));
        }

        std::thread::sleep(Duration::from_millis(
            device_data.interval * 1_000 + OAUTH_POLLING_SAFETY_MARGIN_MS,
        ));

        let token_response = client
            .post(&access_token_url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .json(&serde_json::json!({
                "client_id": CLIENT_ID,
                "device_code": device_data.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }))
            .send()
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = token_response.status();
        let body = token_response
            .text()
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            return Err(AiError::Api(format!(
                "GitHub Copilot device token failed: HTTP {}: {}",
                status, body
            )));
        }

        let token_data: GitHubDeviceTokenResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        if let Some(access_token) = token_data.access_token {
            let auth = GitHubCopilotAuthFile {
                provider: Some("github-copilot".to_string()),
                github_token: access_token.clone(),
                domain: Some(domain.clone()),
                copilot_api_token: None,
                copilot_api_token_expires_at_ms: None,
                copilot_api_base_url: None,
            };
            save_auth_file(&auth_path, &auth)?;

            return Ok(GitHubCopilotDeviceAuth {
                github_token: access_token,
                auth_path,
                domain,
            });
        }

        match token_data.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                std::thread::sleep(Duration::from_secs(token_data.interval.unwrap_or(5) + 5));
                continue;
            }
            Some(other) => {
                return Err(AiError::Api(format!(
                    "GitHub Copilot device flow failed: {}",
                    other
                )));
            }
            None => continue,
        }
    }
}

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

    fn use_custom_base_url(&self) -> bool {
        let base_url = self.config.base_url.trim();
        !base_url.is_empty()
            && base_url != super::providers::github_copilot::spec().default_base_url
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

    fn parse_tool_calls(tool_calls: Option<Vec<GitHubCopilotToolCall>>) -> Vec<ToolCall> {
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

    fn parse_tool_call_deltas(
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
            .map(|message| Self::parse_tool_calls(message.tool_calls.clone()))
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

    async fn exchange_copilot_token(github_token: &str) -> Result<(String, u64, String), AiError> {
        let response = Client::new()
            .get(COPILOT_TOKEN_URL)
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", USER_AGENT)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            return Err(AiError::Api(format!(
                "GitHub Copilot token exchange failed: HTTP {}: {}",
                status, body
            )));
        }

        let response: CopilotTokenExchangeResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;
        let expires_at_ms = parse_token_expiry(&response.expires_at)?;
        let base_url = derive_copilot_api_base_url(&response.token)
            .unwrap_or_else(|| DEFAULT_COPILOT_BASE_URL.to_string());

        Ok((response.token, expires_at_ms, base_url))
    }

    async fn resolve_auth(&self) -> Result<ResolvedCopilotAuth, AiError> {
        if !self.config.api_key.trim().is_empty() {
            let github_token = self.config.api_key.trim();
            let exchanged = Self::exchange_copilot_token(github_token).await;
            let (api_token, base_url) = match exchanged {
                Ok((api_token, _, base_url)) => (api_token, base_url),
                Err(_) => (
                    github_token.to_string(),
                    DEFAULT_COPILOT_BASE_URL.to_string(),
                ),
            };
            return Ok(ResolvedCopilotAuth {
                api_token,
                base_url: if self.use_custom_base_url() {
                    self.config.base_url.clone()
                } else {
                    base_url
                },
            });
        }

        let auth_path = github_copilot_auth_path()?;
        let mut auth = load_auth_file(&auth_path)?;
        if auth.github_token.trim().is_empty() {
            return Err(AiError::Http(format!(
                "GitHub Copilot auth file is missing github_token: {}",
                auth_path.display()
            )));
        }

        if let (Some(api_token), Some(expires_at_ms)) = (
            auth.copilot_api_token.clone(),
            auth.copilot_api_token_expires_at_ms,
        ) {
            if expires_at_ms > now_ms() + TOKEN_REFRESH_SAFETY_WINDOW_MS {
                return Ok(ResolvedCopilotAuth {
                    api_token,
                    base_url: if self.use_custom_base_url() {
                        self.config.base_url.clone()
                    } else {
                        auth.copilot_api_base_url
                            .clone()
                            .unwrap_or_else(|| DEFAULT_COPILOT_BASE_URL.to_string())
                    },
                });
            }
        }

        match Self::exchange_copilot_token(auth.github_token.trim()).await {
            Ok((api_token, expires_at_ms, base_url)) => {
                auth.copilot_api_token = Some(api_token.clone());
                auth.copilot_api_token_expires_at_ms = Some(expires_at_ms);
                auth.copilot_api_base_url = Some(base_url.clone());
                save_auth_file(&auth_path, &auth)?;

                Ok(ResolvedCopilotAuth {
                    api_token,
                    base_url: if self.use_custom_base_url() {
                        self.config.base_url.clone()
                    } else {
                        base_url
                    },
                })
            }
            Err(_) => Ok(ResolvedCopilotAuth {
                api_token: auth.github_token,
                base_url: if self.use_custom_base_url() {
                    self.config.base_url.clone()
                } else {
                    DEFAULT_COPILOT_BASE_URL.to_string()
                },
            }),
        }
    }
}

#[async_trait::async_trait]
impl AiClient for GitHubCopilotClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = self.resolve_auth().await?;
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
            let auth = match GitHubCopilotClient::new(config.clone()).resolve_auth().await {
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
                        let tool_call_deltas = GitHubCopilotClient::parse_tool_call_deltas(
                            choice.delta.tool_calls.clone(),
                        );
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
        let auth = self.resolve_auth().await?;
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
    use super::{GitHubCopilotClient, derive_copilot_api_base_url, parse_token_expiry};
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
