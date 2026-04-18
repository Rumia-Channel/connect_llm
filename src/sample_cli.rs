mod io;
mod settings;
mod streaming;

use self::io::{
    load_input_image, persist_generated_images, print_debug_trace, print_mcp_status,
    print_mcp_tool_executions, print_mcp_tools, print_pending_input_images, print_thinking,
    print_tool_calls, prompt, prompt_default, prompt_multiline,
};
use self::settings::{
    build_ai_config, build_thinking_config, describe_codex_effort, ensure_provider_auth_ready,
    load_mcp_runtime_from_path, normalize_mcp_path, parse_codex_effort, parse_debug_mode,
    parse_stream_mode, parse_thinking_toggle, sanitize_messages_for_request, select_codex_effort,
    select_debug_mode, select_mcp_path, select_model, select_provider, select_stream_mode,
    select_thinking_enabled, temp_client,
};
use self::streaming::{send_mcp_request, send_multimodal_request, send_request};
use connect_llm::{
    ChatRequest, ContentPart, ContextManager, MultimodalChatRequest, RequestMessage,
    set_debug_logging,
};

fn describe_compaction(compaction: &connect_llm::ContextCompaction) -> String {
    if compaction.microcompacted_messages > 0 && compaction.summarized_messages > 0 {
        format!(
            "microcompacted {} messages in {} passes and summarized {} earlier messages ({} -> {} estimated tokens)",
            compaction.microcompacted_messages,
            compaction.microcompaction_passes,
            compaction.summarized_messages,
            compaction.estimated_tokens_before,
            compaction.estimated_tokens_after
        )
    } else if compaction.microcompacted_messages > 0 {
        format!(
            "microcompacted {} messages in {} passes ({} -> {} estimated tokens)",
            compaction.microcompacted_messages,
            compaction.microcompaction_passes,
            compaction.estimated_tokens_before,
            compaction.estimated_tokens_after
        )
    } else {
        format!(
            "compacted {} earlier messages ({} -> {} estimated tokens)",
            compaction.summarized_messages,
            compaction.estimated_tokens_before,
            compaction.estimated_tokens_after
        )
    }
}

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

    let temp_client = temp_client(provider, api_key.clone(), base_url.clone())?;
    let mut model = select_model(provider, temp_client.as_ref()).await?;

    let mut thinking_enabled = select_thinking_enabled(provider)?;
    let mut codex_effort = select_codex_effort(provider)?;
    let mut use_stream = select_stream_mode()?;
    let mut debug_enabled = select_debug_mode()?;
    set_debug_logging(debug_enabled);
    let mut mcp = load_mcp_runtime_from_path(select_mcp_path()?.as_deref()).await?;
    let client =
        provider.create_client(build_ai_config(provider, &api_key, base_url, model.clone()))?;
    let context_manager = ContextManager::default();

    println!();
    println!(
        "Commands: /help, /reset, /model <id>, /thinking <on|off>, /codex-effort <default|minimal|low|medium|high|xhigh>, /stream <on|off>, /debug <on|off>, /image <path-or-url>, /images, /image-clear, /mcp <path|off>, /mcp-status, /mcp-tools, /exit"
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
    println!(
        "MCP: {}",
        mcp.as_ref().map(|(path, _)| path.as_str()).unwrap_or("off")
    );
    if let Some((_, runtime)) = &mcp {
        for server in runtime
            .status()
            .configured_servers
            .into_iter()
            .filter_map(|server| server.last_error.map(|error| (server.label, error)))
        {
            println!("mcp warning> {} {}", server.0, server.1);
        }
    }
    println!();

    let mut messages: Vec<RequestMessage> = Vec::new();
    let mut pending_images = Vec::new();

    loop {
        let input = prompt_multiline("you", "Enter inserts newline. Ctrl+Enter sends.")?;
        let trimmed = input.trim();

        if trimmed.is_empty() && pending_images.is_empty() {
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
            println!("  /image <path-or-url>");
            println!("  /images");
            println!("  /image-clear");
            println!("  /mcp <path|off>");
            println!("  /mcp-status");
            println!("  /mcp-tools");
            println!("  /exit");
            println!("  note: image prompts require MCP off");
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/reset") {
            messages.clear();
            pending_images.clear();
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
                    if let Some(current_path) = mcp.as_ref().map(|(path, _)| path.clone()) {
                        if let Some((_, runtime)) = &mut mcp {
                            runtime.close().await;
                        }
                        match load_mcp_runtime_from_path(Some(&current_path)).await {
                            Ok(next_runtime) => {
                                mcp = next_runtime;
                                println!("MCP reconnected with updated debug mode");
                            }
                            Err(error) => {
                                mcp = None;
                                println!("MCP reconnect failed: {}", error);
                            }
                        }
                    }
                }
                Err(error) => println!("invalid debug mode: {}", error),
            }
            continue;
        }

        if let Some(source) = trimmed.strip_prefix("/image ") {
            if !provider.supports_input_images() {
                println!(
                    "image input unavailable> {} does not support input images in connect_llm 0.2.1",
                    provider.name()
                );
                continue;
            }

            let source = source.trim();
            if matches!(provider, connect_llm::AiProvider::Gemini)
                && (source.starts_with("http://") || source.starts_with("https://"))
            {
                println!(
                    "image input unavailable> {} requires local files or data URLs; remote image URLs are not accepted",
                    provider.name()
                );
                continue;
            }

            match load_input_image(source) {
                Ok(image) => {
                    pending_images.push(image);
                    println!("images queued> {}", pending_images.len());
                }
                Err(error) => println!("image input error> {}", error),
            }
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/images") {
            print_pending_input_images(&pending_images);
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/image-clear") {
            pending_images.clear();
            println!("images cleared");
            continue;
        }

        if let Some(path) = trimmed.strip_prefix("/mcp ") {
            match normalize_mcp_path(path.trim()) {
                Ok(next_path) => match load_mcp_runtime_from_path(next_path.as_deref()).await {
                    Ok(next) => {
                        if let Some((_, runtime)) = &mut mcp {
                            runtime.close().await;
                        }
                        mcp = next;
                        println!(
                            "MCP set to {}",
                            mcp.as_ref().map(|(path, _)| path.as_str()).unwrap_or("off")
                        );
                        if let Some((_, runtime)) = &mcp {
                            for server in
                                runtime.status().configured_servers.into_iter().filter_map(
                                    |server| server.last_error.map(|error| (server.label, error)),
                                )
                            {
                                println!("mcp warning> {} {}", server.0, server.1);
                            }
                        }
                    }
                    Err(error) => println!("invalid mcp config: {}", error),
                },
                Err(error) => println!("invalid mcp path: {}", error),
            }
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/mcp-status") {
            if let Some((path, runtime)) = &mcp {
                println!("mcp path> {}", path);
                print_mcp_status(&runtime.status());
            } else {
                println!("mcp> off");
            }
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/mcp-tools") {
            if let Some((path, runtime)) = &mcp {
                let status = runtime.status();
                println!("mcp path> {}", path);
                print_mcp_tools(&status.exported_tools);
            } else {
                println!("mcp tools> mcp is off");
            }
            continue;
        }

        let mut request_messages = sanitize_messages_for_request(&messages, thinking_enabled);
        let current_user_message = if pending_images.is_empty() {
            RequestMessage::user(input.clone())
        } else {
            let mut parts = Vec::new();
            if !input.is_empty() {
                parts.push(ContentPart::text(input.clone()));
            }
            parts.extend(pending_images.iter().cloned().map(ContentPart::image));
            RequestMessage::user_parts(parts)
        };
        let has_prior_image_history = messages.iter().any(RequestMessage::contains_input_images);
        let use_multimodal_request = current_user_message.contains_input_images();
        request_messages.push(current_user_message.clone());

        let thinking = build_thinking_config(provider, thinking_enabled, codex_effort);

        match if has_prior_image_history {
            Err(connect_llm::AiError::configuration(
                "sample CLI follow-up turns after an image input are not supported yet; use /reset before continuing",
            )
            .with_provider(provider)
            .with_operation("sample_cli"))
        } else if use_multimodal_request {
            if let Some((path, _)) = &mcp {
                Err(connect_llm::AiError::configuration(
                    "sample CLI does not support MCP tool loops with multimodal image history yet",
                )
                .with_provider(provider)
                .with_operation("sample_cli")
                .with_target(path.clone()))
            } else {
                let request = MultimodalChatRequest {
                    model: model.clone(),
                    messages: request_messages,
                    tools: Vec::new(),
                    tool_choice: None,
                    max_tokens: None,
                    temperature: None,
                    system: None,
                    thinking,
                };
                send_multimodal_request(client.as_ref(), request, use_stream, thinking_enabled)
                    .await
                    .map(|response| (response, None, None, None))
            }
        } else if let Some((_, runtime)) = &mut mcp {
            let request = ChatRequest {
                model: model.clone(),
                messages: request_messages
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<_>, _>>()?,
                tools: Vec::new(),
                tool_choice: None,
                max_tokens: None,
                temperature: None,
                system: None,
                thinking,
            };

            if use_stream {
                send_mcp_request(
                    runtime,
                    &context_manager,
                    client.as_ref(),
                    request,
                    thinking_enabled,
                    debug_enabled,
                )
                .await
                .map(|managed| {
                    (
                        managed.response,
                        managed.compaction,
                        Some(managed.messages),
                        Some(managed.tool_executions),
                    )
                })
            } else {
                runtime
                    .chat_with_context_manager(&context_manager, client.as_ref(), request)
                    .await
                    .map(|managed| {
                        (
                            managed.response,
                            managed.compaction,
                            Some(managed.messages),
                            Some(managed.tool_executions),
                        )
                    })
            }
        } else {
            let request = ChatRequest {
                model: model.clone(),
                messages: request_messages
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<_>, _>>()?,
                tools: Vec::new(),
                tool_choice: None,
                max_tokens: None,
                temperature: None,
                system: None,
                thinking,
            };
            let prepared = context_manager
                .prepare_request(client.as_ref(), request)
                .await?;
            if use_stream {
                if let Some(compaction) = &prepared.compaction {
                    println!("context manager> {}", describe_compaction(compaction));
                }
            }
            let prepared_request = prepared.request;
            let prepared_compaction = prepared.compaction.clone();

            if use_stream {
                send_request(client.as_ref(), prepared_request, true, thinking_enabled)
                    .await
                    .map(|response| (response, prepared_compaction, None, None))
            } else {
                context_manager
                    .chat(client.as_ref(), prepared_request)
                    .await
                    .map(|managed| (managed.response, managed.compaction, None, None))
            }
        } {
            Ok((response, compaction, updated_messages, tool_executions)) => {
                println!();
                if !use_stream {
                    println!("assistant> {}", response.content.trim_end());
                    if thinking_enabled {
                        print_thinking(response.thinking.as_ref());
                    }
                    if let Err(error) = persist_generated_images(&response.images) {
                        println!("image output error> {}", error);
                    }
                    print_tool_calls(&response.tool_calls);
                    if let Some(tool_executions) = &tool_executions {
                        print_mcp_tool_executions(tool_executions, debug_enabled);
                    }
                    if let Some(compaction) = &compaction {
                        println!("context manager> {}", describe_compaction(compaction));
                    }
                    println!();
                }
                if debug_enabled {
                    print_debug_trace(response.debug.as_ref());
                }
                if use_stream {
                    if let Err(error) = persist_generated_images(&response.images) {
                        println!("image output error> {}", error);
                    }
                    print_tool_calls(&response.tool_calls);
                    if let Some(compaction) = &compaction {
                        println!("context manager> {}", describe_compaction(compaction));
                    }
                }
                if current_user_message.contains_input_images() {
                    println!(
                        "sample CLI> image turns are single-turn only right now; use /reset before a follow-up"
                    );
                }

                if let Some(updated_messages) = updated_messages {
                    messages = updated_messages.into_iter().map(Into::into).collect();
                } else {
                    if current_user_message.contains_input_images() {
                        pending_images.clear();
                    }
                    messages.push(current_user_message);
                    let mut assistant_message = RequestMessage::assistant(response.content.clone())
                        .with_tool_calls(response.tool_calls.clone());
                    if thinking_enabled {
                        if let Some(thinking) = response.thinking.clone() {
                            assistant_message = assistant_message.with_thinking(thinking);
                        }
                    }
                    messages.push(assistant_message);
                }
            }
            Err(error) => println!("assistant error> {}", error),
        }
    }

    if let Some((_, runtime)) = &mut mcp {
        runtime.close().await;
    }

    Ok(())
}
