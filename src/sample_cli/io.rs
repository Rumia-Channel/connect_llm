use base64::{Engine as _, engine::general_purpose::STANDARD};
use connect_llm::{
    DebugTrace, GeneratedImage, ImageSource, InputImage, McpExportedToolStatus, McpRuntimeStatus,
    McpToolExecution, ThinkingOutput, ToolCall,
};
use crossterm::{
    cursor::{MoveToColumn, MoveUp},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};
use std::{
    collections::HashSet,
    fs,
    io::{self, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

pub(crate) fn print_thinking(thinking: Option<&ThinkingOutput>) {
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

pub(crate) fn print_tool_calls(tool_calls: &[ToolCall]) {
    if tool_calls.is_empty() {
        return;
    }

    for tool_call in tool_calls {
        println!("tools> {} {}", tool_call.name, tool_call.arguments);
    }
}

pub(crate) fn print_mcp_tool_executions(tool_executions: &[McpToolExecution], debug_enabled: bool) {
    if tool_executions.is_empty() {
        return;
    }

    for execution in tool_executions {
        print_mcp_tool_execution(execution, debug_enabled);
    }
}

pub(crate) fn print_mcp_tool_execution(execution: &McpToolExecution, _debug_enabled: bool) {
    let status = if execution.is_error { " error" } else { "" };
    println!(
        "tools> {} {} [{} -> {}.{}{}]",
        execution.tool_name,
        execution.arguments,
        execution.tool_call_id,
        execution.server_label,
        execution.remote_tool_name,
        status
    );
    println!(
        "tool result> {}",
        summarize_json_value(&execution.result, 320)
    );
}

fn summarize_json_value(value: &serde_json::Value, max_chars: usize) -> String {
    let raw = match value {
        serde_json::Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    };
    truncate_for_display(&raw, max_chars)
}

fn truncate_for_display(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }

    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated.push_str(&format!(" ({} chars)", total));
    truncated
}

pub(crate) fn print_debug_trace(debug: Option<&DebugTrace>) {
    let Some(debug) = debug else {
        return;
    };

    if let Some(request) = &debug.request {
        eprintln!("{}", request.trim_end());
    }
    if let Some(response) = &debug.response {
        eprintln!("{}", response.trim_end());
    }
}

pub(crate) fn print_mcp_status(status: &McpRuntimeStatus) {
    println!(
        "mcp> connected={} configured_servers={} connected_servers={}",
        status.connected, status.configured_server_count, status.connected_server_count
    );
    for server in &status.configured_servers {
        let target = server
            .command
            .as_deref()
            .map(str::to_string)
            .or_else(|| server.url.clone())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "mcp server> {} enabled={} connected={} transport={} tools={} target={}",
            server.label,
            server.enabled,
            server.connected,
            server.transport,
            server.exported_tool_count,
            target
        );
        if let Some(error) = &server.last_error {
            println!("mcp error> {} {}", server.label, error);
        }
    }
}

pub(crate) fn print_mcp_tools(tools: &[McpExportedToolStatus]) {
    if tools.is_empty() {
        println!("mcp tools> none");
        return;
    }

    for tool in tools {
        println!(
            "mcp tool> {} -> {}.{}",
            tool.alias, tool.server_label, tool.remote_tool_name
        );
    }
}

pub(crate) fn persist_generated_images(images: &[GeneratedImage]) -> Result<(), io::Error> {
    if images.is_empty() {
        return Ok(());
    }

    let output_dir = PathBuf::from("generated_images");
    fs::create_dir_all(&output_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut seen_image_keys = HashSet::new();
    let mut next_index = 0usize;

    for image in images {
        if !seen_image_keys.insert(image.dedup_key()) {
            continue;
        }

        if let Some(data_base64) = &image.data_base64 {
            let bytes = STANDARD.decode(data_base64).map_err(io::Error::other)?;
            let extension = detect_image_extension(image.mime_type.as_deref(), &bytes);
            let path = output_dir.join(format!("image-{}-{}.{}", timestamp, next_index, extension));
            fs::write(&path, bytes)?;
            println!("image> {}", path.display());
            next_index += 1;
            continue;
        }

        if let Some(url) = &image.url {
            println!("image url> {}", url);
        }
    }

    Ok(())
}

pub(crate) fn load_input_image(source: &str) -> Result<InputImage, io::Error> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "image path or URL is required",
        ));
    }

    if looks_like_remote_image_source(trimmed) {
        return Ok(InputImage::from_url(trimmed.to_string()));
    }

    let path = PathBuf::from(trimmed);
    let bytes = fs::read(&path)?;
    let Some(mime_type) = detect_input_image_mime_type(&path, &bytes) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsupported image type; use png, jpg, webp, gif, bmp, or svg",
        ));
    };

    Ok(InputImage::from_base64(mime_type, STANDARD.encode(bytes)))
}

pub(crate) fn print_pending_input_images(images: &[InputImage]) {
    if images.is_empty() {
        println!("images> none queued");
        return;
    }

    for (index, image) in images.iter().enumerate() {
        match &image.source {
            ImageSource::Base64 {
                mime_type,
                data_base64,
            } => println!(
                "images> [{}] embedded {} ({} base64 chars)",
                index,
                mime_type,
                data_base64.len()
            ),
            ImageSource::Url { url } => println!("images> [{}] {}", index, url),
        }
    }
}

pub(crate) fn prompt(label: &str, hint: &str) -> Result<String, io::Error> {
    print!("{}> ", label);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    if !hint.is_empty() && input.trim().is_empty() {
        let _ = hint;
    }
    Ok(input.trim_end_matches(['\r', '\n']).to_string())
}

pub(crate) fn prompt_default(label: &str, default: &str, hint: &str) -> Result<String, io::Error> {
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

struct RawModeGuard;

impl RawModeGuard {
    fn new() -> Result<Self, io::Error> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

pub(crate) fn prompt_multiline(label: &str, hint: &str) -> Result<String, io::Error> {
    let _ = hint;
    let mut stdout = io::stdout();
    let _raw_mode = RawModeGuard::new()?;

    let mut buffer = String::new();
    let mut rendered_lines = 0usize;
    render_multiline_prompt(&mut stdout, label, &buffer, &mut rendered_lines)?;

    let result = loop {
        match event::read()? {
            Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                match key.code {
                    KeyCode::Enter if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        break Ok(buffer);
                    }
                    KeyCode::Enter => buffer.push('\n'),
                    KeyCode::Backspace => {
                        buffer.pop();
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        break Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "input cancelled",
                        ));
                    }
                    KeyCode::Char(character) => {
                        if !key.modifiers.contains(KeyModifiers::CONTROL)
                            && !key.modifiers.contains(KeyModifiers::ALT)
                        {
                            buffer.push(character);
                        }
                    }
                    KeyCode::Tab => buffer.push('\t'),
                    _ => {}
                }
                render_multiline_prompt(&mut stdout, label, &buffer, &mut rendered_lines)?;
            }
            Event::Paste(text) => {
                buffer.push_str(&text);
                render_multiline_prompt(&mut stdout, label, &buffer, &mut rendered_lines)?;
            }
            _ => {}
        }
    };

    println!();
    result
}

fn render_multiline_prompt(
    stdout: &mut io::Stdout,
    label: &str,
    buffer: &str,
    rendered_lines: &mut usize,
) -> Result<(), io::Error> {
    if *rendered_lines > 0 {
        execute!(stdout, MoveToColumn(0))?;
        for _ in 1..*rendered_lines {
            execute!(stdout, MoveUp(1), MoveToColumn(0))?;
        }
        execute!(stdout, Clear(ClearType::FromCursorDown))?;
    }

    let lines: Vec<&str> = if buffer.is_empty() {
        vec![""]
    } else {
        buffer.split('\n').collect()
    };

    write!(stdout, "{}> {}", label, lines[0])?;
    for line in lines.iter().skip(1) {
        write!(stdout, "\r\n..  {}", line)?;
    }
    stdout.flush()?;
    *rendered_lines = lines.len();
    Ok(())
}

fn detect_image_extension(mime_type: Option<&str>, bytes: &[u8]) -> &'static str {
    if let Some(mime_type) = mime_type {
        let mime_type = mime_type.to_ascii_lowercase();
        if mime_type.contains("png") {
            return "png";
        }
        if mime_type.contains("jpeg") || mime_type.contains("jpg") {
            return "jpg";
        }
        if mime_type.contains("webp") {
            return "webp";
        }
        if mime_type.contains("gif") {
            return "gif";
        }
    }

    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return "png";
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "jpg";
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "webp";
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return "gif";
    }

    "bin"
}

fn looks_like_remote_image_source(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://") || value.starts_with("data:")
}

fn detect_input_image_mime_type(path: &std::path::Path, bytes: &[u8]) -> Option<&'static str> {
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "png" => return Some("image/png"),
            "jpg" | "jpeg" => return Some("image/jpeg"),
            "webp" => return Some("image/webp"),
            "gif" => return Some("image/gif"),
            "bmp" => return Some("image/bmp"),
            "svg" => return Some("image/svg+xml"),
            _ => {}
        }
    }

    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }
    if bytes.starts_with(b"<svg") || bytes.windows(4).any(|window| window == b"<svg") {
        return Some("image/svg+xml");
    }

    None
}
