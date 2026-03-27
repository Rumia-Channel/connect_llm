use conect_llm::{
    AiConfig, AiProvider, ChatRequest, DebugTrace, Message, ThinkingConfig, ThinkingEffort,
    ThinkingOutput, github_copilot_auth_path, login_github_copilot_via_device,
    login_openai_codex_via_browser, openai_codex_auth_path, set_debug_logging,
};
use futures_util::StreamExt;
use std::io::{self, Write};

const PROVIDERS: [AiProvider; 11] = [
    AiProvider::Sakura,
    AiProvider::Anthropic,
    AiProvider::GitHubCopilot,
    AiProvider::OpenAi,
    AiProvider::OpenAiCodex,
    AiProvider::Kimi,
    AiProvider::KimiCoding,
    AiProvider::ZAi,
    AiProvider::ZAiCoding,
    AiProvider::GoogleAiStudio,
    AiProvider::Gemini,
];

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {}", error);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("conect_llm sample chat");
    println!("API key is only kept in this process memory.");
    println!();

    let provider = select_provider()?;
    let base_url = prompt_default(
        "base URL",
        provider.default_base_url(),
        "Press Enter to use the provider default.",
    )?;
    let api_key_hint = if provider == AiProvider::OpenAiCodex {
        "Press Enter to use CODEX_HOME/auth.json or ~/.codex/auth.json."
    } else if provider == AiProvider::GitHubCopilot {
        "Press Enter to use COPILOT_HOME/auth.json or ~/.copilot/auth.json."
    } else {
        "Typed visibly, stored only in memory for this run."
    };
    let api_key = prompt("API key", api_key_hint)?;

    ensure_provider_auth_ready(provider, &api_key).await?;

    let temp_client = provider.create_client(AiConfig {
        api_key: api_key.clone(),
        base_url: base_url.clone(),
        model: provider.default_model().to_string(),
    });
    let mut model = select_model(provider, temp_client.as_ref()).await?;

    let mut thinking_enabled = select_thinking_enabled(provider)?;
    let mut codex_effort = select_codex_effort(provider)?;
    let mut use_stream = select_stream_mode()?;
    let mut debug_enabled = select_debug_mode()?;
    set_debug_logging(debug_enabled);
    let client = provider.create_client(AiConfig {
        api_key,
        base_url,
        model: model.clone(),
    });

    println!();
    println!(
        "Commands: /help, /reset, /model <id>, /thinking <on|off>, /codex-effort <default|minimal|low|medium|high|xhigh>, /stream <on|off>, /debug <on|off>, /exit"
    );
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);
    println!(
        "Thinking output: {}",
        if thinking_enabled { "on" } else { "off" }
    );
    println!("Codex effort: {}", describe_codex_effort(codex_effort));
    println!("Stream: {}", if use_stream { "on" } else { "off" });
    println!("Debug: {}", if debug_enabled { "on" } else { "off" });
    println!();

    let mut messages: Vec<Message> = Vec::new();

    loop {
        let input = prompt("you", "Type a prompt or /exit.")?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/exit") {
            break;
        }

        if trimmed.eq_ignore_ascii_case("/help") {
            println!("Commands:");
            println!("  /reset");
            println!("  /model <id>");
            println!("  /thinking <on|off>");
            println!("  /codex-effort <default|minimal|low|medium|high|xhigh>");
            println!("  /stream <on|off>");
            println!("  /debug <on|off>");
            println!("  /exit");
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/reset") {
            messages.clear();
            println!("history cleared");
            continue;
        }

        if let Some(next_model) = trimmed.strip_prefix("/model ") {
            let next_model = next_model.trim();
            if next_model.is_empty() {
                println!("model unchanged");
            } else {
                model = next_model.to_string();
                println!("model set to {}", model);
            }
            continue;
        }

        if let Some(mode) = trimmed.strip_prefix("/thinking ") {
            match parse_thinking_toggle(mode.trim()) {
                Ok(next) => {
                    thinking_enabled = next;
                    println!(
                        "thinking output set to {}",
                        if thinking_enabled { "on" } else { "off" }
                    );
                }
                Err(error) => println!("invalid thinking mode: {}", error),
            }
            continue;
        }

        if let Some(level) = trimmed.strip_prefix("/codex-effort ") {
            match parse_codex_effort(level.trim()) {
                Ok(next) => {
                    codex_effort = next;
                    println!(
                        "codex effort set to {}",
                        describe_codex_effort(codex_effort)
                    );
                }
                Err(error) => println!("invalid codex effort: {}", error),
            }
            continue;
        }

        if let Some(mode) = trimmed.strip_prefix("/stream ") {
            match parse_stream_mode(mode.trim()) {
                Ok(next) => {
                    use_stream = next;
                    println!(
                        "stream mode set to {}",
                        if use_stream { "on" } else { "off" }
                    );
                }
                Err(error) => println!("invalid stream mode: {}", error),
            }
            continue;
        }

        if let Some(mode) = trimmed.strip_prefix("/debug ") {
            match parse_debug_mode(mode.trim()) {
                Ok(next) => {
                    debug_enabled = next;
                    set_debug_logging(debug_enabled);
                    println!(
                        "debug mode set to {}",
                        if debug_enabled { "on" } else { "off" }
                    );
                }
                Err(error) => println!("invalid debug mode: {}", error),
            }
            continue;
        }

        let mut request_messages = sanitize_messages_for_request(&messages, thinking_enabled);
        request_messages.push(Message {
            role: "user".to_string(),
            content: input.clone(),
            thinking: None,
        });

        messages.push(Message {
            role: "user".to_string(),
            content: input,
            thinking: None,
        });

        let request = ChatRequest {
            model: model.clone(),
            messages: request_messages,
            max_tokens: Some(4096),
            temperature: None,
            system: None,
            thinking: build_thinking_config(provider, thinking_enabled, codex_effort),
        };

        match send_request(client.as_ref(), request, use_stream, thinking_enabled).await {
            Ok(response) => {
                println!();
                if !use_stream {
                    println!("assistant> {}", response.content.trim_end());
                    if thinking_enabled {
                        print_thinking(response.thinking.as_ref());
                    }
                    println!();
                }

                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response.content,
                    thinking: if thinking_enabled {
                        response.thinking
                    } else {
                        None
                    },
                });
            }
            Err(error) => {
                println!("assistant error> {}", error);
            }
        }
    }

    Ok(())
}

async fn send_request(
    client: &dyn conect_llm::AiClient,
    request: ChatRequest,
    use_stream: bool,
    include_thinking: bool,
) -> Result<conect_llm::ChatResponse, conect_llm::AiError> {
    if !use_stream {
        let mut response = client.chat(request).await?;
        if !include_thinking {
            response.thinking = None;
        }
        return Ok(response);
    }

    let model = request.model.clone();
    let mut stream = client.chat_stream(request);
    let mut content = String::new();
    let mut thinking_text = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut printed_assistant_prefix = false;
    let mut printed_thinking_prefix = false;
    let mut debug_request: Option<String> = None;
    let mut debug_responses: Vec<String> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        if let Some(debug) = chunk.debug {
            if debug_request.is_none() {
                debug_request = debug.request;
            }
            if let Some(response) = debug.response {
                debug_responses.push(response);
            }
        }

        if !chunk.delta.is_empty() {
            if !printed_assistant_prefix {
                print!("\nassistant> ");
                printed_assistant_prefix = true;
            }
            print!("{}", chunk.delta);
            io::stdout()
                .flush()
                .map_err(|error| conect_llm::AiError::Http(error.to_string()))?;
            content.push_str(&chunk.delta);
        }

        if include_thinking {
            if let Some(thinking_delta) = chunk.thinking_delta {
                if !printed_thinking_prefix {
                    print!("\nthinking> ");
                    printed_thinking_prefix = true;
                }
                print!("{}", thinking_delta);
                io::stdout()
                    .flush()
                    .map_err(|error| conect_llm::AiError::Http(error.to_string()))?;
                thinking_text.push_str(&thinking_delta);
            }
            if let Some(signature) = chunk.thinking_signature {
                thinking_signature = Some(signature);
            }
        }

        if chunk.done {
            break;
        }
    }

    if printed_assistant_prefix || printed_thinking_prefix {
        println!();
    }

    Ok(conect_llm::ChatResponse {
        id: "stream".to_string(),
        content,
        model,
        usage: conect_llm::Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        thinking: if include_thinking && !thinking_text.is_empty() {
            Some(ThinkingOutput {
                text: Some(thinking_text),
                signature: thinking_signature,
                redacted: None,
            })
        } else if include_thinking && thinking_signature.is_some() {
            Some(ThinkingOutput {
                text: None,
                signature: thinking_signature,
                redacted: None,
            })
        } else {
            None
        },
        debug: if debug_request.is_some() || !debug_responses.is_empty() {
            Some(DebugTrace {
                request: debug_request,
                response: if debug_responses.is_empty() {
                    None
                } else {
                    Some(debug_responses.join("\n"))
                },
            })
        } else {
            None
        },
    })
}

async fn ensure_provider_auth_ready(
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

fn select_provider() -> Result<AiProvider, Box<dyn std::error::Error>> {
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

async fn select_model(
    provider: AiProvider,
    client: &dyn conect_llm::AiClient,
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

fn select_thinking_enabled(provider: AiProvider) -> Result<bool, Box<dyn std::error::Error>> {
    let default_value =
        if provider.supports_thinking_output() || provider.supports_thinking_config() {
            "on"
        } else {
            "off"
        };
    let input = prompt_default("thinking", default_value, "Available: on, off.")?;
    parse_thinking_toggle(&input).map_err(|error| error.into())
}

fn select_codex_effort(
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

fn select_stream_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let input = prompt_default("stream", "off", "Available: on, off.")?;
    parse_stream_mode(&input).map_err(|error| error.into())
}

fn select_debug_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let input = prompt_default("debug", "off", "Available: on, off.")?;
    parse_debug_mode(&input).map_err(|error| error.into())
}

fn parse_thinking_toggle(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "on" | "true" | "yes" | "1" => Ok(true),
        "off" | "false" | "no" | "0" => Ok(false),
        _ => Err("expected on or off".to_string()),
    }
}

fn parse_codex_effort(value: &str) -> Result<Option<ThinkingEffort>, String> {
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

fn parse_stream_mode(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "false" | "no" | "0" => Ok(false),
        "on" | "true" | "yes" | "1" => Ok(true),
        _ => Err("expected on or off".to_string()),
    }
}

fn parse_debug_mode(value: &str) -> Result<bool, String> {
    parse_stream_mode(value)
}

fn describe_codex_effort(effort: Option<ThinkingEffort>) -> String {
    match effort {
        None => "default".to_string(),
        Some(ThinkingEffort::Minimal) => "minimal".to_string(),
        Some(ThinkingEffort::Low) => "low".to_string(),
        Some(ThinkingEffort::Medium) => "medium".to_string(),
        Some(ThinkingEffort::High) => "high".to_string(),
        Some(ThinkingEffort::XHigh) => "xhigh".to_string(),
    }
}

fn build_thinking_config(
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

fn sanitize_messages_for_request(messages: &[Message], include_thinking: bool) -> Vec<Message> {
    messages
        .iter()
        .cloned()
        .map(|mut message| {
            if !include_thinking {
                message.thinking = None;
            }
            message
        })
        .collect()
}

fn print_thinking(thinking: Option<&ThinkingOutput>) {
    let Some(thinking) = thinking else {
        return;
    };

    if let Some(text) = &thinking.text {
        println!("thinking> {}", text.trim_end());
    }
    if let Some(signature) = &thinking.signature {
        println!("thinking signature> {}", signature);
    }
    if let Some(redacted) = &thinking.redacted {
        println!("thinking redacted> {}", redacted);
    }
}

fn prompt(label: &str, hint: &str) -> Result<String, io::Error> {
    print!("{}> ", label);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    if !hint.is_empty() && input.trim().is_empty() {
        let _ = hint;
    }
    Ok(input.trim_end_matches(['\r', '\n']).to_string())
}

fn prompt_default(label: &str, default: &str, hint: &str) -> Result<String, io::Error> {
    print!("{} [{}]> ", label, default);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let value = input.trim_end_matches(['\r', '\n']).trim();
    if value.is_empty() {
        if !hint.is_empty() {
            let _ = hint;
        }
        Ok(default.to_string())
    } else {
        Ok(value.to_string())
    }
}
