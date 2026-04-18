use crate::sample_cli::io::{prompt, prompt_default};
use connect_llm::{
    AiAuth, AiConfig, AiProvider, McpBridge, McpConfig, McpRuntime, ThinkingConfig, ThinkingEffort,
    github_copilot_auth_path, login_github_copilot_via_device, login_openai_codex_via_browser,
    openai_codex_auth_path,
};
use std::{path::Path, sync::Arc};

fn config_auth_for_provider(provider: AiProvider, api_key: &str) -> AiAuth {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return AiAuth::None;
    }

    match provider {
        AiProvider::Anthropic | AiProvider::Gemini => AiAuth::ApiKey(api_key.to_string()),
        _ => AiAuth::BearerToken(api_key.to_string()),
    }
}

pub(crate) fn build_ai_config(
    provider: AiProvider,
    api_key: impl AsRef<str>,
    base_url: impl Into<String>,
    model: impl Into<String>,
) -> AiConfig {
    AiConfig::new(provider)
        .with_auth(config_auth_for_provider(provider, api_key.as_ref()))
        .with_base_url(base_url)
        .with_default_model(model)
}

pub(crate) const PROVIDERS: [AiProvider; 12] = [
    AiProvider::Sakura,
    AiProvider::Anthropic,
    AiProvider::GitHubCopilot,
    AiProvider::Grok,
    AiProvider::OpenAi,
    AiProvider::OpenAiCodex,
    AiProvider::Kimi,
    AiProvider::KimiCoding,
    AiProvider::ZAi,
    AiProvider::ZAiCoding,
    AiProvider::GoogleAiStudio,
    AiProvider::Gemini,
];

pub(crate) async fn ensure_provider_auth_ready(
    provider: AiProvider,
    api_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if provider == AiProvider::OpenAiCodex && api_key.trim().is_empty() {
        let needs_login = openai_codex_auth_path()
            .map(|path| !path.exists())
            .unwrap_or(true);
        if needs_login {
            println!("No saved OpenAI Codex auth found. Starting browser login.");
            tokio::task::spawn_blocking(|| login_openai_codex_via_browser(Default::default()))
                .await??;
        }
    }

    if provider == AiProvider::GitHubCopilot && api_key.trim().is_empty() {
        let needs_login = github_copilot_auth_path()
            .map(|path| !path.exists())
            .unwrap_or(true);
        if needs_login {
            println!("No saved GitHub Copilot auth found. Starting device login.");
            tokio::task::spawn_blocking(|| login_github_copilot_via_device(Default::default()))
                .await??;
        }
    }

    Ok(())
}

pub(crate) fn select_provider() -> Result<AiProvider, Box<dyn std::error::Error>> {
    println!("Providers:");
    for (index, provider) in PROVIDERS.iter().enumerate() {
        println!(
            "  {}: {} (default model: {})",
            index,
            provider.name(),
            provider.default_model()
        );
    }

    loop {
        let input = prompt("provider", "Index or name.")?;
        let trimmed = input.trim();

        if let Ok(index) = trimmed.parse::<usize>() {
            if let Some(provider) = PROVIDERS.get(index) {
                return Ok(*provider);
            }
        }

        if let Some(provider) = PROVIDERS
            .iter()
            .copied()
            .find(|provider| provider.name().eq_ignore_ascii_case(trimmed))
        {
            return Ok(provider);
        }

        println!("unknown provider");
    }
}

pub(crate) async fn select_model(
    provider: AiProvider,
    client: &dyn connect_llm::AiClient,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("Fetching models...");

    let default_model = provider.default_model().to_string();
    let models = match client.list_models().await {
        Ok(models) if !models.is_empty() => models,
        Ok(_) => {
            println!("No models were returned by the provider. Falling back to manual input.");
            return prompt_default(
                "model",
                &default_model,
                "Press Enter to use the provider default.",
            )
            .map_err(|error| error.into());
        }
        Err(error) => {
            println!("Could not fetch model list: {}", error);
            return prompt_default(
                "model",
                &default_model,
                "Press Enter to use the provider default.",
            )
            .map_err(|error| error.into());
        }
    };

    println!("Models:");
    for (index, model) in models.iter().enumerate() {
        if model == &default_model {
            println!("  {}: {} (default)", index, model);
        } else {
            println!("  {}: {}", index, model);
        }
    }

    loop {
        let input = prompt_default("model", &default_model, "Index or model id.")?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Ok(default_model.clone());
        }

        if let Ok(index) = trimmed.parse::<usize>() {
            if let Some(model) = models.get(index) {
                return Ok(model.clone());
            }
        }

        if trimmed.eq_ignore_ascii_case(&default_model) {
            return Ok(default_model.clone());
        }

        if let Some(model) = models
            .iter()
            .find(|model| model.eq_ignore_ascii_case(trimmed))
        {
            return Ok(model.clone());
        }

        println!("unknown model");
    }
}

pub(crate) fn select_thinking_enabled(
    provider: AiProvider,
) -> Result<bool, Box<dyn std::error::Error>> {
    let default_value =
        if provider.supports_thinking_output() || provider.supports_thinking_config() {
            "on"
        } else {
            "off"
        };
    let input = prompt_default("thinking", default_value, "Available: on, off.")?;
    parse_thinking_toggle(&input).map_err(|error| error.into())
}

pub(crate) fn select_codex_effort(
    provider: AiProvider,
) -> Result<Option<ThinkingEffort>, Box<dyn std::error::Error>> {
    let default_value =
        if provider == AiProvider::OpenAiCodex || provider == AiProvider::GitHubCopilot {
            "medium"
        } else {
            "default"
        };
    let input = prompt_default(
        "codex effort",
        default_value,
        "Available: default, minimal, low, medium, high, xhigh.",
    )?;
    parse_codex_effort(&input).map_err(|error| error.into())
}

pub(crate) fn select_stream_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let input = prompt_default("stream", "off", "Available: on, off.")?;
    parse_stream_mode(&input).map_err(|error| error.into())
}

pub(crate) fn select_debug_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let input = prompt_default("debug", "off", "Available: on, off.")?;
    parse_debug_mode(&input).map_err(|error| error.into())
}

pub(crate) fn select_mcp_path() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let input = prompt_default(
        "mcp.json",
        "",
        "Optional path to an mcp.json file. Leave empty to disable MCP.",
    )?;
    normalize_mcp_path(&input).map_err(|error| error.into())
}

pub(crate) fn parse_thinking_toggle(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "on" | "true" | "yes" | "1" => Ok(true),
        "off" | "false" | "no" | "0" => Ok(false),
        _ => Err("expected on or off".to_string()),
    }
}

pub(crate) fn parse_codex_effort(value: &str) -> Result<Option<ThinkingEffort>, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "default" | "auto" => Ok(None),
        "minimal" => Ok(Some(ThinkingEffort::Minimal)),
        "low" => Ok(Some(ThinkingEffort::Low)),
        "medium" | "adaptive" => Ok(Some(ThinkingEffort::Medium)),
        "high" => Ok(Some(ThinkingEffort::High)),
        "xhigh" | "x-high" | "extra-high" => Ok(Some(ThinkingEffort::XHigh)),
        _ => Err("expected default, minimal, low, medium, high, or xhigh".to_string()),
    }
}

pub(crate) fn parse_stream_mode(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "false" | "no" | "0" => Ok(false),
        "on" | "true" | "yes" | "1" => Ok(true),
        _ => Err("expected on or off".to_string()),
    }
}

pub(crate) fn parse_debug_mode(value: &str) -> Result<bool, String> {
    parse_stream_mode(value)
}

pub(crate) fn normalize_mcp_path(value: &str) -> Result<Option<String>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") || trimmed == "-" {
        return Ok(None);
    }

    let mut normalized = trimmed;
    loop {
        let unquoted = normalized
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                normalized
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            });
        let Some(next) = unquoted else {
            break;
        };
        normalized = next.trim();
    }

    if normalized.is_empty() {
        return Ok(None);
    }

    let path = Path::new(normalized);
    let normalized_path = if path.is_dir() {
        path.join("mcp.json").to_string_lossy().into_owned()
    } else {
        normalized.to_string()
    };

    Ok(Some(normalized_path))
}

pub(crate) async fn load_mcp_runtime_from_path(
    path: Option<&str>,
) -> Result<Option<(String, McpRuntime)>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let config = McpConfig::from_path(path)?;
    let runtime = McpBridge::new(config).connect().await?;
    Ok(Some((path.to_string(), runtime)))
}

pub(crate) fn describe_codex_effort(effort: Option<ThinkingEffort>) -> String {
    match effort {
        None => "default".to_string(),
        Some(ThinkingEffort::Minimal) => "minimal".to_string(),
        Some(ThinkingEffort::Low) => "low".to_string(),
        Some(ThinkingEffort::Medium) => "medium".to_string(),
        Some(ThinkingEffort::High) => "high".to_string(),
        Some(ThinkingEffort::XHigh) => "xhigh".to_string(),
    }
}

pub(crate) fn build_thinking_config(
    provider: AiProvider,
    thinking_enabled: bool,
    codex_effort: Option<ThinkingEffort>,
) -> Option<ThinkingConfig> {
    if !thinking_enabled {
        return if provider.supports_thinking_config() {
            Some(ThinkingConfig::disabled())
        } else {
            None
        };
    }

    let mut thinking = ThinkingConfig::enabled();
    if provider == AiProvider::OpenAiCodex || provider == AiProvider::GitHubCopilot {
        thinking.effort = codex_effort;
    }
    Some(thinking)
}

pub(crate) fn sanitize_messages_for_request(
    messages: &[connect_llm::RequestMessage],
    include_thinking: bool,
) -> Vec<connect_llm::RequestMessage> {
    messages
        .iter()
        .cloned()
        .map(|mut message| {
            if !include_thinking {
                message.clear_thinking();
            }
            message
        })
        .collect()
}

pub(crate) fn temp_client(
    provider: AiProvider,
    api_key: String,
    base_url: String,
) -> Result<Arc<dyn connect_llm::AiClient>, connect_llm::AiError> {
    provider.create_client(build_ai_config(
        provider,
        api_key,
        base_url,
        provider.default_model().to_string(),
    ))
}
