use conect_llm::{
    AiConfig, AiProvider, ChatRequest, Message, ThinkingConfig, ThinkingEffort, ThinkingOutput,
};
use futures_util::StreamExt;
use std::io::{self, Write};

const PROVIDERS: [AiProvider; 10] = [
    AiProvider::Sakura,
    AiProvider::Anthropic,
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
    let mut model = prompt_default(
        "model",
        provider.default_model(),
        "Press Enter to use the provider default.",
    )?;
    let api_key_hint = if provider == AiProvider::OpenAiCodex {
        "Press Enter to use CODEX_HOME/auth.json or ~/.codex/auth.json."
    } else {
        "Typed visibly, stored only in memory for this run."
    };
    let api_key = prompt("API key", api_key_hint)?;

    let mut thinking = select_thinking_config(provider)?;
    let mut use_stream = select_stream_mode()?;
    let client = provider.create_client(AiConfig {
        api_key,
        base_url,
        model: model.clone(),
    });

    println!();
    println!(
        "Commands: /help, /reset, /model <id>, /thinking <off|minimal|low|medium|high|xhigh>, /stream <on|off>, /exit"
    );
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);
    println!("Thinking config: {}", describe_thinking(thinking.as_ref()));
    println!("Stream: {}", if use_stream { "on" } else { "off" });
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
            println!("  /thinking <off|minimal|low|medium|high|xhigh>");
            println!("  /stream <on|off>");
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

        if let Some(level) = trimmed.strip_prefix("/thinking ") {
            match parse_thinking_level(level.trim()) {
                Ok(next) => {
                    thinking = next;
                    println!(
                        "thinking config set to {}",
                        describe_thinking(thinking.as_ref())
                    );
                }
                Err(error) => println!("invalid thinking level: {}", error),
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

        messages.push(Message {
            role: "user".to_string(),
            content: input,
            thinking: None,
        });

        let request = ChatRequest {
            model: model.clone(),
            messages: messages.clone(),
            max_tokens: Some(4096),
            temperature: None,
            system: None,
            thinking: thinking.clone(),
        };

        match send_request(client.as_ref(), request, use_stream).await {
            Ok(response) => {
                println!();
                if !use_stream {
                    println!("assistant> {}", response.content.trim_end());
                    print_thinking(response.thinking.as_ref());
                    println!();
                }

                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response.content,
                    thinking: response.thinking,
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
) -> Result<conect_llm::ChatResponse, conect_llm::AiError> {
    if !use_stream {
        return client.chat(request).await;
    }

    let model = request.model.clone();
    let mut stream = client.chat_stream(request);
    let mut content = String::new();
    let mut thinking_text = String::new();
    let mut printed_assistant_prefix = false;
    let mut printed_thinking_prefix = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

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
        thinking: if thinking_text.is_empty() {
            None
        } else {
            Some(ThinkingOutput {
                text: Some(thinking_text),
                signature: None,
                redacted: None,
            })
        },
    })
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

fn select_thinking_config(
    provider: AiProvider,
) -> Result<Option<ThinkingConfig>, Box<dyn std::error::Error>> {
    if !provider.supports_thinking_config() {
        println!("Thinking config is not advertised for this provider.");
        return Ok(None);
    }

    let default_level = if provider == AiProvider::OpenAiCodex {
        "medium"
    } else {
        "off"
    };
    let input = prompt_default(
        "thinking",
        default_level,
        "Available: off, minimal, low, medium, high, xhigh.",
    )?;
    parse_thinking_level(&input).map_err(|error| error.into())
}

fn select_stream_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let input = prompt_default("stream", "off", "Available: on, off.")?;
    parse_stream_mode(&input).map_err(|error| error.into())
}

fn parse_thinking_level(value: &str) -> Result<Option<ThinkingConfig>, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "off" | "none" => Ok(None),
        "minimal" => Ok(Some(ThinkingConfig::enabled_with_effort(
            ThinkingEffort::Minimal,
        ))),
        "low" => Ok(Some(ThinkingConfig::enabled_with_effort(
            ThinkingEffort::Low,
        ))),
        "medium" | "adaptive" => Ok(Some(ThinkingConfig::enabled_with_effort(
            ThinkingEffort::Medium,
        ))),
        "high" => Ok(Some(ThinkingConfig::enabled_with_effort(
            ThinkingEffort::High,
        ))),
        "xhigh" | "x-high" | "extra-high" => Ok(Some(ThinkingConfig::enabled_with_effort(
            ThinkingEffort::XHigh,
        ))),
        _ => Err("expected off, minimal, low, medium, high, or xhigh".to_string()),
    }
}

fn parse_stream_mode(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "false" | "no" | "0" => Ok(false),
        "on" | "true" | "yes" | "1" => Ok(true),
        _ => Err("expected on or off".to_string()),
    }
}

fn describe_thinking(thinking: Option<&ThinkingConfig>) -> String {
    let Some(thinking) = thinking else {
        return "off".to_string();
    };

    if !thinking.enabled {
        return "off".to_string();
    }

    match thinking.effort.unwrap_or(ThinkingEffort::Medium) {
        ThinkingEffort::Minimal => "minimal".to_string(),
        ThinkingEffort::Low => "low".to_string(),
        ThinkingEffort::Medium => "medium".to_string(),
        ThinkingEffort::High => "high".to_string(),
        ThinkingEffort::XHigh => "xhigh".to_string(),
    }
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
