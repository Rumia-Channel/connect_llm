#![allow(dead_code)]

use super::{
    AiClient, AiConfig, AiError, ChatRequest, ChatResponse, StreamChunk, ThinkingConfig,
    ThinkingEffort, ThinkingOutput, Usage,
};
use futures_util::StreamExt;
use rand::{RngCore, rngs::OsRng};
use reqwest::{Client, Url, blocking::Client as BlockingClient};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER: &str = "https://auth.openai.com";
const DEFAULT_CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
const DEFAULT_CALLBACK_PORT: u16 = 1455;
const DEFAULT_AUTH_TIMEOUT_SECS: u64 = 300;
const REFRESH_SAFETY_WINDOW_MS: u64 = 30_000;

const BROWSER_AUTH_SUCCESS_HTML: &str = r#"<!doctype html>
<html>
  <head>
    <title>conect_llm - Codex Authorization Successful</title>
    <style>
      body {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #131010;
        color: #f1ecec;
      }
      .container {
        text-align: center;
        padding: 2rem;
      }
      p {
        color: #b7b1b1;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Authorization Successful</h1>
      <p>You can close this window and return to your app.</p>
    </div>
    <script>
      setTimeout(() => window.close(), 1500)
    </script>
  </body>
</html>"#;

#[derive(Debug, Clone)]
pub struct OpenAiCodexBrowserAuthOptions {
    pub callback_port: u16,
    pub timeout: Duration,
    pub auth_path: Option<PathBuf>,
    pub open_browser: bool,
}

impl Default for OpenAiCodexBrowserAuthOptions {
    fn default() -> Self {
        Self {
            callback_port: DEFAULT_CALLBACK_PORT,
            timeout: Duration::from_secs(DEFAULT_AUTH_TIMEOUT_SECS),
            auth_path: None,
            open_browser: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCodexBrowserAuth {
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: Option<String>,
    pub account_id: Option<String>,
    pub expires_at_ms: u64,
    pub auth_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiCodexRequest {
    model: String,
    messages: Vec<OpenAiCodexMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAiCodexReasoningRequest>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiCodexReasoningRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexResponse {
    id: String,
    choices: Vec<OpenAiCodexChoice>,
    model: String,
    #[serde(default)]
    usage: Option<OpenAiCodexUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexChoice {
    message: OpenAiCodexMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct OpenAiCodexUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexStreamResponse {
    #[allow(dead_code)]
    id: String,
    choices: Vec<OpenAiCodexStreamChoice>,
    #[allow(dead_code)]
    model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexStreamChoice {
    delta: OpenAiCodexDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiCodexDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
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

#[derive(Debug, Clone, Deserialize)]
struct CodexRefreshResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Clone)]
struct ResolvedCodexAuth {
    access_token: String,
    account_id: Option<String>,
}

#[derive(Debug, Clone)]
struct LoadedCodexAuthFile {
    path: PathBuf,
    document: Value,
    access_token: String,
    refresh_token: String,
    id_token: Option<String>,
    account_id: Option<String>,
    expires_at_ms: Option<u64>,
}

fn format_error_detail(status: reqwest::StatusCode, detail: &OpenAiCodexErrorDetail) -> String {
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
    if let Ok(error) = serde_json::from_str::<OpenAiCodexError>(body) {
        return AiError::Api(format_error_detail(status, &error.error));
    }

    AiError::Api(format!("HTTP {}: {}", status, body))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn base64_url_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::new();
    let mut index = 0usize;

    while index + 3 <= bytes.len() {
        let a = bytes[index];
        let b = bytes[index + 1];
        let c = bytes[index + 2];
        output.push(TABLE[(a >> 2) as usize] as char);
        output.push(TABLE[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
        output.push(TABLE[(((b & 0x0f) << 2) | (c >> 6)) as usize] as char);
        output.push(TABLE[(c & 0x3f) as usize] as char);
        index += 3;
    }

    match bytes.len().saturating_sub(index) {
        1 => {
            let a = bytes[index];
            output.push(TABLE[(a >> 2) as usize] as char);
            output.push(TABLE[((a & 0x03) << 4) as usize] as char);
        }
        2 => {
            let a = bytes[index];
            let b = bytes[index + 1];
            output.push(TABLE[(a >> 2) as usize] as char);
            output.push(TABLE[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
            output.push(TABLE[((b & 0x0f) << 2) as usize] as char);
        }
        _ => {}
    }

    output
}

fn generate_random_url_safe_string(byte_len: usize) -> String {
    let mut bytes = vec![0u8; byte_len];
    OsRng.fill_bytes(&mut bytes);
    base64_url_encode(&bytes)
}

fn generate_pkce_verifier() -> String {
    generate_random_url_safe_string(32)
}

fn generate_pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64_url_encode(digest.as_slice())
}

fn build_authorize_url(redirect_uri: &str, verifier: &str, state: &str) -> Result<String, AiError> {
    let challenge = generate_pkce_challenge(verifier);
    let mut url = Url::parse(&format!("{}/oauth/authorize", ISSUER))
        .map_err(|error| AiError::Parse(error.to_string()))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", "openid profile email offline_access")
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("state", state)
        .append_pair("originator", "conect_llm");
    Ok(url.to_string())
}

fn browser_auth_error_html(message: &str) -> String {
    format!(
        r#"<!doctype html>
<html>
  <head>
    <title>conect_llm - Codex Authorization Failed</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #131010;
        color: #f1ecec;
      }}
      .container {{
        text-align: center;
        padding: 2rem;
      }}
      .error {{
        color: #ff917b;
        font-family: monospace;
        margin-top: 1rem;
        padding: 1rem;
        background: #3c140d;
        border-radius: 0.5rem;
        max-width: 40rem;
        word-break: break-word;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Authorization Failed</h1>
      <div class="error">{}</div>
    </div>
  </body>
</html>"#,
        message
    )
}

fn send_browser_callback_response(stream: &mut std::net::TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
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

fn wait_for_browser_callback(
    listener: TcpListener,
    expected_state: String,
    timeout: Duration,
) -> Result<String, AiError> {
    let (sender, receiver) = mpsc::channel();

    thread::spawn(move || {
        let accept_result = listener.accept();
        let result = match accept_result {
            Ok((mut stream, _)) => {
                let mut buffer = [0u8; 8192];
                let read_len = stream.read(&mut buffer);
                match read_len {
                    Ok(len) if len > 0 => {
                        let request = String::from_utf8_lossy(&buffer[..len]);
                        let request_line = request.lines().next().unwrap_or_default();
                        let target = request_line.split_whitespace().nth(1).unwrap_or("/");

                        match Url::parse(&format!("http://localhost{}", target)) {
                            Ok(url) => {
                                let state = url
                                    .query_pairs()
                                    .find(|(key, _)| key == "state")
                                    .map(|(_, value)| value.to_string());
                                if state.as_deref() != Some(expected_state.as_str()) {
                                    let message = "State mismatch in OAuth callback.";
                                    send_browser_callback_response(
                                        &mut stream,
                                        "400 Bad Request",
                                        &browser_auth_error_html(message),
                                    );
                                    Err(AiError::Api(message.to_string()))
                                } else if let Some(error) = url
                                    .query_pairs()
                                    .find(|(key, _)| key == "error")
                                    .map(|(_, value)| value.to_string())
                                {
                                    send_browser_callback_response(
                                        &mut stream,
                                        "400 Bad Request",
                                        &browser_auth_error_html(&error),
                                    );
                                    Err(AiError::Api(format!(
                                        "OAuth authorization failed: {}",
                                        error
                                    )))
                                } else if let Some(code) = url
                                    .query_pairs()
                                    .find(|(key, _)| key == "code")
                                    .map(|(_, value)| value.to_string())
                                {
                                    send_browser_callback_response(
                                        &mut stream,
                                        "200 OK",
                                        BROWSER_AUTH_SUCCESS_HTML,
                                    );
                                    Ok(code)
                                } else {
                                    let message =
                                        "OAuth callback did not contain an authorization code.";
                                    send_browser_callback_response(
                                        &mut stream,
                                        "400 Bad Request",
                                        &browser_auth_error_html(message),
                                    );
                                    Err(AiError::Api(message.to_string()))
                                }
                            }
                            Err(error) => {
                                let message = format!("Invalid OAuth callback URL: {}", error);
                                send_browser_callback_response(
                                    &mut stream,
                                    "400 Bad Request",
                                    &browser_auth_error_html(&message),
                                );
                                Err(AiError::Parse(message))
                            }
                        }
                    }
                    Ok(_) => Err(AiError::Api(
                        "OAuth callback connection closed without a request.".to_string(),
                    )),
                    Err(error) => Err(AiError::Http(error.to_string())),
                }
            }
            Err(error) => Err(AiError::Http(error.to_string())),
        };

        let _ = sender.send(result);
    });

    receiver.recv_timeout(timeout).map_err(|_| {
        AiError::Api("Timed out waiting for the browser OAuth callback.".to_string())
    })?
}

fn decode_jwt_payload(token: &str) -> Option<Value> {
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    let _signature = parts.next()?;
    let bytes = base64_url_decode(payload)?;
    serde_json::from_slice(&bytes).ok()
}

fn decode_jwt_exp_ms(token: &str) -> Option<u64> {
    decode_jwt_payload(token)?
        .get("exp")?
        .as_u64()
        .map(|exp| exp.saturating_mul(1000))
}

fn base64_url_decode(input: &str) -> Option<Vec<u8>> {
    let mut normalized = input.replace('-', "+").replace('_', "/");
    while normalized.len() % 4 != 0 {
        normalized.push('=');
    }

    base64_decode(&normalized)
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut chunk = [0u8; 4];
    let mut chunk_len = 0usize;

    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }

        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'\r' | b'\n' | b'\t' | b' ' => continue,
            _ => return None,
        };

        chunk[chunk_len] = value;
        chunk_len += 1;

        if chunk_len == 4 {
            output.push((chunk[0] << 2) | (chunk[1] >> 4));
            output.push((chunk[1] << 4) | (chunk[2] >> 2));
            output.push((chunk[2] << 6) | chunk[3]);
            chunk_len = 0;
        }
    }

    match chunk_len {
        0 => Some(output),
        2 => {
            output.push((chunk[0] << 2) | (chunk[1] >> 4));
            Some(output)
        }
        3 => {
            output.push((chunk[0] << 2) | (chunk[1] >> 4));
            output.push((chunk[1] << 4) | (chunk[2] >> 2));
            Some(output)
        }
        _ => None,
    }
}

fn extract_account_id_from_claims(claims: &Value) -> Option<String> {
    if let Some(account_id) = claims.get("chatgpt_account_id").and_then(Value::as_str) {
        if !account_id.is_empty() {
            return Some(account_id.to_string());
        }
    }

    if let Some(account_id) = claims
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
    {
        if !account_id.is_empty() {
            return Some(account_id.to_string());
        }
    }

    claims
        .get("organizations")
        .and_then(Value::as_array)
        .and_then(|organizations| organizations.first())
        .and_then(|organization| organization.get("id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn extract_account_id_from_tokens(
    id_token: Option<&str>,
    access_token: Option<&str>,
) -> Option<String> {
    id_token
        .and_then(decode_jwt_payload)
        .as_ref()
        .and_then(extract_account_id_from_claims)
        .or_else(|| {
            access_token
                .and_then(decode_jwt_payload)
                .as_ref()
                .and_then(extract_account_id_from_claims)
        })
}

fn resolve_codex_home() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CODEX_HOME") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }

    if let Ok(path) = std::env::var("HOME") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed).join(".codex"));
        }
    }

    if let Ok(path) = std::env::var("USERPROFILE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed).join(".codex"));
        }
    }

    None
}

fn resolve_codex_auth_path() -> Result<PathBuf, AiError> {
    resolve_codex_auth_path_with_override(None)
}

fn resolve_codex_auth_path_with_override(auth_path: Option<PathBuf>) -> Result<PathBuf, AiError> {
    if let Some(path) = auth_path {
        return Ok(path);
    }

    let home = resolve_codex_home().ok_or_else(|| {
        AiError::Api("Could not resolve CODEX_HOME or user home directory.".to_string())
    })?;
    Ok(home.join("auth.json"))
}

fn read_codex_auth_file(path: &Path) -> Result<LoadedCodexAuthFile, AiError> {
    let body = fs::read_to_string(path).map_err(|error| {
        AiError::Api(format!(
            "Failed to read Codex auth file at {}: {}",
            path.display(),
            error
        ))
    })?;
    let document: Value =
        serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

    let tokens = document
        .get("tokens")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            AiError::Api("Codex auth file does not contain a tokens object.".to_string())
        })?;

    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AiError::Api("Codex auth file does not contain access_token.".to_string()))?
        .to_string();

    let refresh_token = tokens
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AiError::Api("Codex auth file does not contain refresh_token.".to_string()))?
        .to_string();

    let id_token = tokens
        .get("id_token")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    let account_id = tokens
        .get("account_id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| extract_account_id_from_tokens(id_token.as_deref(), Some(&access_token)));

    Ok(LoadedCodexAuthFile {
        path: path.to_path_buf(),
        document,
        access_token: access_token.clone(),
        refresh_token,
        id_token,
        account_id,
        expires_at_ms: decode_jwt_exp_ms(&access_token),
    })
}

fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    value
        .as_object_mut()
        .expect("value was just initialized as object")
}

fn write_codex_auth_file(
    loaded: &LoadedCodexAuthFile,
    refreshed: &CodexRefreshResponse,
    account_id: Option<&str>,
) -> Result<(), AiError> {
    let mut document = loaded.document.clone();
    let tokens = ensure_object(
        ensure_object(&mut document)
            .entry("tokens".to_string())
            .or_insert_with(|| Value::Object(Map::new())),
    );

    tokens.insert(
        "access_token".to_string(),
        Value::String(refreshed.access_token.clone()),
    );
    tokens.insert(
        "refresh_token".to_string(),
        Value::String(refreshed.refresh_token.clone()),
    );

    if let Some(id_token) = &refreshed.id_token {
        tokens.insert("id_token".to_string(), Value::String(id_token.clone()));
    }

    if let Some(account_id) = account_id {
        tokens.insert(
            "account_id".to_string(),
            Value::String(account_id.to_string()),
        );
    }

    document["last_refresh"] = json!(now_ms());

    let text = serde_json::to_string_pretty(&document)
        .map_err(|error| AiError::Parse(error.to_string()))?;
    fs::write(&loaded.path, text).map_err(|error| {
        AiError::Api(format!(
            "Failed to update Codex auth file at {}: {}",
            loaded.path.display(),
            error
        ))
    })
}

fn persist_codex_auth_file(
    path: &Path,
    existing: Option<Value>,
    refreshed: &CodexRefreshResponse,
    account_id: Option<&str>,
) -> Result<(), AiError> {
    let loaded = LoadedCodexAuthFile {
        path: path.to_path_buf(),
        document: existing.unwrap_or_else(|| json!({ "auth_mode": "chatgpt" })),
        access_token: refreshed.access_token.clone(),
        refresh_token: refreshed.refresh_token.clone(),
        id_token: refreshed.id_token.clone(),
        account_id: account_id.map(ToString::to_string),
        expires_at_ms: refreshed
            .expires_in
            .map(|expires_in| now_ms().saturating_add(expires_in.saturating_mul(1000))),
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            AiError::Api(format!(
                "Failed to create Codex auth directory at {}: {}",
                parent.display(),
                error
            ))
        })?;
    }

    write_codex_auth_file(&loaded, refreshed, account_id)
}

fn exchange_code_for_tokens_blocking(
    code: &str,
    redirect_uri: &str,
    verifier: &str,
) -> Result<CodexRefreshResponse, AiError> {
    let body = format!(
        "grant_type=authorization_code&code={}&redirect_uri={}&client_id={}&code_verifier={}",
        percent_encode_component(code),
        percent_encode_component(redirect_uri),
        percent_encode_component(CLIENT_ID),
        percent_encode_component(verifier),
    );

    let response = BlockingClient::new()
        .post(format!("{}/oauth/token", ISSUER))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .map_err(|error| AiError::Http(error.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .map_err(|error| AiError::Http(error.to_string()))?;

    if !status.is_success() {
        return Err(api_error_from_response(status, &body));
    }

    serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))
}

pub fn login_openai_codex_via_browser(
    options: OpenAiCodexBrowserAuthOptions,
) -> Result<OpenAiCodexBrowserAuth, AiError> {
    let callback_port = options.callback_port;
    let redirect_uri = format!("http://localhost:{}/auth/callback", callback_port);
    let verifier = generate_pkce_verifier();
    let state = generate_random_url_safe_string(24);
    let authorize_url = build_authorize_url(&redirect_uri, &verifier, &state)?;
    let listener = TcpListener::bind(("127.0.0.1", callback_port)).map_err(|error| {
        AiError::Api(format!(
            "Failed to bind OpenAI Codex OAuth callback on localhost:{}: {}",
            callback_port, error
        ))
    })?;

    if options.open_browser {
        open_url_in_browser(&authorize_url)?;
    } else {
        return Err(AiError::Api(format!(
            "Browser auto-open is disabled. Open this URL manually and implement a custom callback flow: {}",
            authorize_url
        )));
    }

    let code = wait_for_browser_callback(listener, state, options.timeout)?;
    let refreshed = exchange_code_for_tokens_blocking(&code, &redirect_uri, &verifier)?;
    let account_id = extract_account_id_from_tokens(
        refreshed.id_token.as_deref(),
        Some(&refreshed.access_token),
    );
    let auth_path = resolve_codex_auth_path_with_override(options.auth_path)?;
    let existing_document = fs::read_to_string(&auth_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());

    persist_codex_auth_file(
        &auth_path,
        existing_document,
        &refreshed,
        account_id.as_deref(),
    )?;

    Ok(OpenAiCodexBrowserAuth {
        access_token: refreshed.access_token,
        refresh_token: refreshed.refresh_token,
        id_token: refreshed.id_token,
        account_id,
        expires_at_ms: now_ms().saturating_add(refreshed.expires_in.unwrap_or(3600) * 1000),
        auth_path,
    })
}

pub struct OpenAiCodexClient {
    client: Client,
    config: AiConfig,
}

impl OpenAiCodexClient {
    pub fn new(config: AiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn endpoint_url(base_url: &str) -> String {
        let trimmed = base_url.trim_end_matches('/');
        if trimmed.is_empty() {
            return DEFAULT_CODEX_ENDPOINT.to_string();
        }
        if trimmed.ends_with("/codex/responses") {
            trimmed.to_string()
        } else {
            format!("{}/codex/responses", trimmed)
        }
    }

    fn convert_reasoning_config(
        thinking: Option<&ThinkingConfig>,
    ) -> Option<OpenAiCodexReasoningRequest> {
        let thinking = thinking?;

        if !thinking.enabled {
            return Some(OpenAiCodexReasoningRequest {
                effort: Some("none"),
            });
        }

        let effort = match thinking.effort.unwrap_or(ThinkingEffort::Medium) {
            ThinkingEffort::Minimal => "minimal",
            ThinkingEffort::Low => "low",
            ThinkingEffort::Medium => "medium",
            ThinkingEffort::High => "high",
            ThinkingEffort::XHigh => "xhigh",
        };

        Some(OpenAiCodexReasoningRequest {
            effort: Some(effort),
        })
    }

    fn convert_request(request: ChatRequest, stream: bool) -> OpenAiCodexRequest {
        let ChatRequest {
            model,
            messages: request_messages,
            max_tokens,
            temperature,
            system,
            thinking,
        } = request;

        let mut messages = Vec::new();

        if let Some(system) = system {
            messages.push(OpenAiCodexMessage {
                role: "system".to_string(),
                content: system,
                reasoning_content: None,
            });
        }

        for message in request_messages {
            let super::Message {
                role,
                content,
                thinking,
            } = message;
            messages.push(OpenAiCodexMessage {
                role,
                content,
                reasoning_content: thinking.and_then(|thinking| thinking.text),
            });
        }

        OpenAiCodexRequest {
            model,
            messages,
            max_tokens,
            temperature,
            reasoning: Self::convert_reasoning_config(thinking.as_ref()),
            stream,
        }
    }

    fn convert_response(response: OpenAiCodexResponse) -> ChatResponse {
        let thinking = response
            .choices
            .first()
            .and_then(|choice| choice.message.reasoning_content.clone())
            .map(|text| ThinkingOutput {
                text: Some(text),
                signature: None,
                redacted: None,
            });

        let content = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
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
        }
    }

    async fn refresh_access_token(
        &self,
        refresh_token: &str,
    ) -> Result<CodexRefreshResponse, AiError> {
        let request_body = format!(
            "grant_type=refresh_token&refresh_token={}&client_id={}",
            percent_encode_component(refresh_token),
            percent_encode_component(CLIENT_ID),
        );
        let response = self
            .client
            .post(format!("{}/oauth/token", ISSUER))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(request_body)
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))
    }

    async fn resolve_auth(&self) -> Result<ResolvedCodexAuth, AiError> {
        let configured_token = self.config.api_key.trim();
        if !configured_token.is_empty() {
            return Ok(ResolvedCodexAuth {
                access_token: configured_token.to_string(),
                account_id: extract_account_id_from_tokens(None, Some(configured_token)),
            });
        }

        let path = resolve_codex_auth_path()?;
        let loaded = read_codex_auth_file(&path)?;
        let expires_at = loaded.expires_at_ms.unwrap_or(u64::MAX);

        if expires_at > now_ms().saturating_add(REFRESH_SAFETY_WINDOW_MS) {
            return Ok(ResolvedCodexAuth {
                access_token: loaded.access_token,
                account_id: loaded.account_id,
            });
        }

        let refreshed = self.refresh_access_token(&loaded.refresh_token).await?;
        let account_id = extract_account_id_from_tokens(
            refreshed.id_token.as_deref().or(loaded.id_token.as_deref()),
            Some(&refreshed.access_token),
        )
        .or(loaded.account_id.clone());

        write_codex_auth_file(&loaded, &refreshed, account_id.as_deref())?;

        Ok(ResolvedCodexAuth {
            access_token: refreshed.access_token,
            account_id,
        })
    }
}

fn percent_encode_component(input: &str) -> String {
    let mut output = String::new();

    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(byte as char)
            }
            _ => output.push_str(&format!("%{:02X}", byte)),
        }
    }

    output
}

#[async_trait::async_trait]
impl AiClient for OpenAiCodexClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
        let auth = self.resolve_auth().await?;
        let url = Self::endpoint_url(&self.config.base_url);
        let request = Self::convert_request(request, false);

        let mut builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.access_token))
            .header("Content-Type", "application/json")
            .json(&request);

        if let Some(account_id) = auth.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }

        let response = builder
            .send()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| AiError::Http(error.to_string()))?;

        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let response: OpenAiCodexResponse =
            serde_json::from_str(&body).map_err(|error| AiError::Parse(error.to_string()))?;

        Ok(Self::convert_response(response))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<StreamChunk, AiError>> {
        let client = self.client.clone();
        let config = self.config.clone();
        let request = Self::convert_request(request, true);

        let stream = async_stream::stream! {
            let auth = match OpenAiCodexClient::new(config.clone()).resolve_auth().await {
                Ok(auth) => auth,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let mut builder = client
                .post(OpenAiCodexClient::endpoint_url(&config.base_url))
                .header("Authorization", format!("Bearer {}", auth.access_token))
                .header("Content-Type", "application/json")
                .json(&request);

            if let Some(account_id) = auth.account_id {
                builder = builder.header("ChatGPT-Account-Id", account_id);
            }

            let response = builder.send().await;
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

                    let data = &line[6..];
                    if data == "[DONE]" {
                        yield Ok(StreamChunk {
                            delta: String::new(),
                            thinking_delta: None,
                            thinking_signature: None,
                            done: true,
                        });
                        return;
                    }

                    let response: OpenAiCodexStreamResponse = match serde_json::from_str(data) {
                        Ok(response) => response,
                        Err(_) => continue,
                    };

                    if let Some(choice) = response.choices.first() {
                        let delta = choice.delta.content.clone().unwrap_or_default();
                        let thinking_delta = choice.delta.reasoning_content.clone();
                        let done = choice.finish_reason.is_some();

                        yield Ok(StreamChunk {
                            delta,
                            thinking_delta,
                            thinking_signature: None,
                            done,
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
                done: true,
            });
        };

        futures_util::StreamExt::boxed(stream)
    }

    fn config(&self) -> &AiConfig {
        &self.config
    }

    async fn list_models(&self) -> Result<Vec<String>, AiError> {
        Ok(vec![
            "gpt-5.4".to_string(),
            "gpt-5.4-mini".to_string(),
            "gpt-5.3-codex".to_string(),
            "gpt-5.3-codex-spark".to_string(),
            "gpt-5.2".to_string(),
            "gpt-5.2-codex".to_string(),
            "gpt-5.1-codex".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::{OpenAiCodexClient, base64_url_decode, extract_account_id_from_tokens};
    use crate::ai::{ChatRequest, Message, ThinkingConfig, ThinkingEffort};

    #[test]
    fn decodes_base64_url_without_padding() {
        let decoded = base64_url_decode("eyJmb28iOiJiYXIifQ").expect("decoded");
        assert_eq!(
            String::from_utf8(decoded).expect("utf8"),
            r#"{"foo":"bar"}"#
        );
    }

    #[test]
    fn codex_request_uses_medium_reasoning_when_thinking_is_enabled() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                thinking: None,
            }],
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: Some(ThinkingConfig::enabled()),
        };

        let converted = OpenAiCodexClient::convert_request(request, false);
        assert_eq!(
            converted.reasoning.and_then(|reasoning| reasoning.effort),
            Some("medium")
        );
    }

    #[test]
    fn codex_request_uses_explicit_reasoning_effort() {
        let request = ChatRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                thinking: None,
            }],
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: Some(ThinkingConfig::enabled_with_effort(ThinkingEffort::XHigh)),
        };

        let converted = OpenAiCodexClient::convert_request(request, false);
        assert_eq!(
            converted.reasoning.and_then(|reasoning| reasoning.effort),
            Some("xhigh")
        );
    }

    #[test]
    fn extracts_account_id_from_access_token_claims() {
        let token = "eyJhbGciOiJub25lIn0.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoib3JnXzEyMyJ9fQ.";
        assert_eq!(
            extract_account_id_from_tokens(None, Some(token)).as_deref(),
            Some("org_123")
        );
    }
}
