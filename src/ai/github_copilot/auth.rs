use super::protocol::api_error_from_response;
use crate::ai::{
    AiConfig, AiError,
    auth_common::{now_ms, open_url_in_browser},
};
use reqwest::{Client, blocking::Client as BlockingClient};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
pub(super) struct ResolvedCopilotAuth {
    pub api_token: String,
    pub base_url: String,
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

fn home_dir() -> Result<PathBuf, AiError> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| AiError::api("Could not resolve user home directory.".to_string()))
}

pub fn github_copilot_auth_path() -> Result<PathBuf, AiError> {
    Ok(home_dir()?.join(".copilot").join("auth.json"))
}

fn normalize_domain(domain: &str) -> String {
    let trimmed = domain.trim();
    if trimmed.is_empty() {
        DEFAULT_DOMAIN.to_string()
    } else {
        trimmed.to_string()
    }
}

pub(super) fn derive_copilot_api_base_url(token: &str) -> Option<String> {
    token
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("proxy-ep="))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| format!("https://api.{}", value.trim_start_matches("proxy.")))
}

pub(super) fn parse_token_expiry(expires_at: &serde_json::Value) -> Result<u64, AiError> {
    let raw = expires_at
        .as_u64()
        .or_else(|| {
            expires_at
                .as_str()
                .and_then(|value| value.parse::<u64>().ok())
        })
        .ok_or_else(|| {
            AiError::parse("GitHub Copilot token expiry was not a number.".to_string())
        })?;

    Ok(if raw >= 100_000_000_000 {
        raw
    } else {
        raw.saturating_mul(1000)
    })
}

fn load_auth_file(path: &Path) -> Result<GitHubCopilotAuthFile, AiError> {
    let body = fs::read_to_string(path).map_err(|error| {
        AiError::api(format!(
            "Failed to read GitHub Copilot auth file at {}: {}",
            path.display(),
            error
        ))
    })?;
    serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))
}

fn save_auth_file(path: &Path, auth: &GitHubCopilotAuthFile) -> Result<(), AiError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            AiError::api(format!(
                "Failed to create GitHub Copilot auth directory at {}: {}",
                parent.display(),
                error
            ))
        })?;
    }

    let body =
        serde_json::to_string_pretty(auth).map_err(|error| AiError::parse(error.to_string()))?;
    let mut file = fs::File::create(path).map_err(|error| {
        AiError::api(format!(
            "Failed to write GitHub Copilot auth file at {}: {}",
            path.display(),
            error
        ))
    })?;
    file.write_all(body.as_bytes())
        .map_err(|error| AiError::api(error.to_string()))
}

pub fn login_github_copilot_via_device(
    options: GitHubCopilotDeviceAuthOptions,
) -> Result<GitHubCopilotDeviceAuth, AiError> {
    let domain = normalize_domain(&options.domain);
    let auth_path = options
        .auth_path
        .clone()
        .unwrap_or(github_copilot_auth_path()?);

    let device_code_response = BlockingClient::new()
        .post(format!("https://{}/login/device/code", domain))
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "client_id={}&scope={}",
            percent_encode_component(CLIENT_ID),
            percent_encode_component("read:user"),
        ))
        .send()
        .map_err(|error| AiError::http(error.to_string()))?;

    let status = device_code_response.status();
    let body = device_code_response
        .text()
        .map_err(|error| AiError::http(error.to_string()))?;
    if !status.is_success() {
        return Err(api_error_from_response(status, &body));
    }

    let device_code: GitHubDeviceCodeResponse =
        serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;

    println!("GitHub Copilot device login");
    println!("Visit: {}", device_code.verification_uri);
    println!("Code: {}", device_code.user_code);

    if options.open_browser {
        let _ = open_url_in_browser(&device_code.verification_uri);
    }

    println!("Waiting for authorization...");

    let started_at = std::time::Instant::now();
    let timeout = options.timeout;
    let mut interval_secs = device_code.interval.max(1);

    loop {
        if started_at.elapsed() > timeout {
            return Err(AiError::api(
                "Timed out waiting for GitHub device authorization.".to_string(),
            ));
        }

        std::thread::sleep(Duration::from_secs(interval_secs));

        let token_response = BlockingClient::new()
            .post(format!("https://{}/login/oauth/access_token", domain))
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(format!(
                "client_id={}&device_code={}&grant_type={}",
                percent_encode_component(CLIENT_ID),
                percent_encode_component(device_code.device_code.as_str()),
                percent_encode_component("urn:ietf:params:oauth:grant-type:device_code"),
            ))
            .send()
            .map_err(|error| AiError::http(error.to_string()))?;

        let status = token_response.status();
        let body = token_response
            .text()
            .map_err(|error| AiError::http(error.to_string()))?;
        if !status.is_success() {
            return Err(api_error_from_response(status, &body));
        }

        let token_data: GitHubDeviceTokenResponse =
            serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;

        if let Some(access_token) = token_data.access_token {
            let auth_file = GitHubCopilotAuthFile {
                provider: Some("github_copilot".to_string()),
                github_token: access_token.clone(),
                domain: Some(domain.clone()),
                copilot_api_token: None,
                copilot_api_token_expires_at_ms: None,
                copilot_api_base_url: None,
            };
            save_auth_file(&auth_path, &auth_file)?;
            return Ok(GitHubCopilotDeviceAuth {
                github_token: access_token,
                auth_path,
                domain,
            });
        }

        match token_data.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                interval_secs = token_data.interval.unwrap_or(interval_secs) + 5;
                continue;
            }
            Some(other) => {
                return Err(AiError::api(format!(
                    "GitHub Copilot device flow failed: {}",
                    other
                )));
            }
            None => continue,
        }
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
        .map_err(|error| AiError::http(error.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AiError::http(error.to_string()))?;

    if !status.is_success() {
        return Err(AiError::api(format!(
            "GitHub Copilot token exchange failed: HTTP {}: {}",
            status, body
        )));
    }

    let response: CopilotTokenExchangeResponse =
        serde_json::from_str(&body).map_err(|error| AiError::parse(error.to_string()))?;
    let expires_at_ms = parse_token_expiry(&response.expires_at)?;
    let base_url = derive_copilot_api_base_url(&response.token)
        .unwrap_or_else(|| DEFAULT_COPILOT_BASE_URL.to_string());

    Ok((response.token, expires_at_ms, base_url))
}

pub(super) async fn resolve_auth(config: &AiConfig) -> Result<ResolvedCopilotAuth, AiError> {
    if let Some(github_token) = config
        .bearer_token()
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let exchanged = exchange_copilot_token(github_token).await;
        let (api_token, base_url) = match exchanged {
            Ok((api_token, _, base_url)) => (api_token, base_url),
            Err(_) => (
                github_token.to_string(),
                derive_copilot_api_base_url(github_token)
                    .unwrap_or_else(|| config.base_url().trim().to_string()),
            ),
        };

        let base_url = if base_url.is_empty() {
            DEFAULT_COPILOT_BASE_URL.to_string()
        } else {
            base_url
        };

        return Ok(ResolvedCopilotAuth {
            api_token,
            base_url,
        });
    }

    let path = github_copilot_auth_path()?;
    let mut auth = load_auth_file(&path)?;
    let github_token = auth.github_token.trim();
    if github_token.is_empty() {
        return Err(
            AiError::auth("GitHub Copilot auth file does not contain github_token.")
                .with_operation("resolve_auth")
                .with_target(path.display().to_string()),
        );
    }

    let configured_base_url = config.base_url().trim();
    if let (Some(api_token), Some(expires_at_ms)) = (
        auth.copilot_api_token.clone(),
        auth.copilot_api_token_expires_at_ms,
    ) {
        if expires_at_ms > now_ms().saturating_add(TOKEN_REFRESH_SAFETY_WINDOW_MS) {
            let base_url = if configured_base_url.is_empty() {
                auth.copilot_api_base_url
                    .clone()
                    .unwrap_or_else(|| DEFAULT_COPILOT_BASE_URL.to_string())
            } else {
                configured_base_url.to_string()
            };

            return Ok(ResolvedCopilotAuth {
                api_token,
                base_url,
            });
        }
    }

    let exchanged = exchange_copilot_token(github_token).await;
    let (api_token, expires_at_ms, base_url) = match exchanged {
        Ok((api_token, expires_at_ms, base_url)) => (api_token, expires_at_ms, base_url),
        Err(_) => (
            github_token.to_string(),
            now_ms().saturating_add(OAUTH_POLLING_SAFETY_MARGIN_MS),
            derive_copilot_api_base_url(github_token)
                .or_else(|| auth.copilot_api_base_url.clone())
                .unwrap_or_else(|| DEFAULT_COPILOT_BASE_URL.to_string()),
        ),
    };

    auth.copilot_api_token = Some(api_token.clone());
    auth.copilot_api_token_expires_at_ms = Some(expires_at_ms);
    auth.copilot_api_base_url = Some(base_url.clone());
    save_auth_file(&path, &auth)?;

    let base_url = if configured_base_url.is_empty() {
        base_url
    } else {
        configured_base_url.to_string()
    };

    Ok(ResolvedCopilotAuth {
        api_token,
        base_url,
    })
}
