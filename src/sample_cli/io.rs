use base64::{Engine as _, engine::general_purpose::STANDARD};
use connect_llm::{DebugTrace, GeneratedImage, ThinkingOutput, ToolCall};
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
        println!("tool call> {} {}", tool_call.name, tool_call.arguments);
    }
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

pub(crate) fn prompt_multiline(label: &str, hint: &str) -> Result<String, io::Error> {
    let _ = hint;
    let mut stdout = io::stdout();
    enable_raw_mode()?;

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

    disable_raw_mode()?;
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
