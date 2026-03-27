use connect_llm::{DebugTrace, ThinkingOutput, ToolCall};
use crossterm::{
    cursor::{MoveToColumn, MoveUp},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};
use std::io::{self, Write};

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
