#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextWindowConfig {
    pub max_chars: usize,
    pub overlap_chars: usize,
}

impl Default for TextWindowConfig {
    fn default() -> Self {
        Self {
            max_chars: 24_000,
            overlap_chars: 512,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextWindow {
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
}

pub fn split_text_into_windows(text: &str, config: TextWindowConfig) -> Vec<TextWindow> {
    if text.is_empty() {
        return Vec::new();
    }

    let max_chars = config.max_chars.max(1);
    let overlap_chars = config.overlap_chars.min(max_chars.saturating_sub(1));
    let step = max_chars.saturating_sub(overlap_chars).max(1);

    let chars: Vec<char> = text.chars().collect();
    let mut windows = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;

    while start < chars.len() {
        let end = (start + max_chars).min(chars.len());
        windows.push(TextWindow {
            index,
            start_char: start,
            end_char: end,
            text: chars[start..end].iter().collect(),
        });

        if end == chars.len() {
            break;
        }

        start = start.saturating_add(step);
        index += 1;
    }

    windows
}
