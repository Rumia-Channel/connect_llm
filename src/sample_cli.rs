mod io;
mod settings;
mod streaming;

use self::io::{print_thinking, print_tool_calls, prompt, prompt_default, prompt_multiline};
use self::settings::{
    build_thinking_config, describe_codex_effort, ensure_provider_auth_ready, parse_codex_effort,
    parse_debug_mode, parse_stream_mode, parse_thinking_toggle, sanitize_messages_for_request,
    select_codex_effort, select_debug_mode, select_model, select_provider, select_stream_mode,
    select_thinking_enabled, temp_client,
};
use self::streaming::send_request;
use connect_llm::{AiConfig, ChatRequest, ContextManager, Message, set_debug_logging};

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("connect_llm sample chat");
    println!("API key is only kept in this process memory.");
    println!("Chat input uses Enter for newline and Ctrl+Enter to send.");
    println!();

    let provider = select_provider()?;
    let base_url = prompt_default(
        "base URL",
        provider.default_base_url(),
        "Press Enter to use the provider default.",
    )?;
    let api_key_hint = if provider == connect_llm::AiProvider::OpenAiCodex {
        "Press Enter to use CODEX_HOME/auth.json or ~/.codex/auth.json."
    } else if provider == connect_llm::AiProvider::GitHubCopilot {
        "Press Enter to use COPILOT_HOME/auth.json or ~/.copilot/auth.json."
    } else {
        "Typed visibly, stored only in memory for this run."
    };
    let api_key = prompt("API key", api_key_hint)?;

    ensure_provider_auth_ready(provider, &api_key).await?;

    let temp_client = temp_client(provider, api_key.clone(), base_url.clone());
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
    let context_manager = ContextManager::default();

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
        let input = prompt_multiline("you", "Enter inserts newline. Ctrl+Enter sends.")?;
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
        request_messages.push(Message::user(input.clone()));
        messages.push(Message::user(input));

        let request = ChatRequest {
            model: model.clone(),
            messages: request_messages,
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            system: None,
            thinking: build_thinking_config(provider, thinking_enabled, codex_effort),
        };

        let prepared = context_manager
            .prepare_request(client.as_ref(), request)
            .await?;
        if use_stream {
            if let Some(compaction) = &prepared.compaction {
                println!(
                    "context manager> compacted {} earlier messages ({} -> {} estimated tokens)",
                    compaction.summarized_messages,
                    compaction.estimated_tokens_before,
                    compaction.estimated_tokens_after
                );
            }
        }
        let prepared_request = prepared.request;
        let prepared_compaction = prepared.compaction.clone();

        match if use_stream {
            send_request(client.as_ref(), prepared_request, true, thinking_enabled)
                .await
                .map(|response| (response, prepared_compaction))
        } else {
            context_manager
                .chat(client.as_ref(), prepared_request)
                .await
                .map(|managed| (managed.response, managed.compaction))
        } {
            Ok((response, compaction)) => {
                println!();
                if !use_stream {
                    println!("assistant> {}", response.content.trim_end());
                    if thinking_enabled {
                        print_thinking(response.thinking.as_ref());
                    }
                    print_tool_calls(&response.tool_calls);
                    if let Some(compaction) = &compaction {
                        println!(
                            "context manager> compacted {} earlier messages ({} -> {} estimated tokens)",
                            compaction.summarized_messages,
                            compaction.estimated_tokens_before,
                            compaction.estimated_tokens_after
                        );
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
                    tool_calls: response.tool_calls,
                    tool_call_id: None,
                    tool_name: None,
                    tool_result: None,
                    tool_error: None,
                });
            }
            Err(error) => println!("assistant error> {}", error),
        }
    }

    Ok(())
}
