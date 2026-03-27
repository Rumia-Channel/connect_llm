#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelContextLimits {
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

pub fn resolve_model_context_limits(base_url: &str, model: &str) -> ModelContextLimits {
    let base_url = base_url.trim_end_matches('/').to_ascii_lowercase();
    let model = model.to_ascii_lowercase();

    if base_url.contains("api.kimi.com/coding") {
        return ModelContextLimits {
            context_window: Some(262_144),
            max_output_tokens: Some(32_768),
        };
    }

    if base_url.contains("api.anthropic.com") {
        return ModelContextLimits {
            context_window: Some(200_000),
            max_output_tokens: Some(match model.as_str() {
                model if model.starts_with("claude-opus-4-1") => 32_000,
                model if model.starts_with("claude-opus-4") => 32_000,
                model if model.starts_with("claude-sonnet-4") => 64_000,
                model if model.starts_with("claude-3-7-sonnet") => 64_000,
                model if model.starts_with("claude-3-5-sonnet") => 8_192,
                model if model.starts_with("claude-3-5-haiku") => 8_192,
                model if model.starts_with("claude-3-haiku") => 4_096,
                _ => 8_192,
            }),
        };
    }

    if base_url.contains("generativelanguage.googleapis.com") || model.contains("gemini-") {
        return ModelContextLimits {
            context_window: Some(1_048_576),
            max_output_tokens: Some(65_536),
        };
    }

    if model == "gpt-4.1" || model.starts_with("gpt-4.1-") {
        return ModelContextLimits {
            context_window: Some(1_047_576),
            max_output_tokens: Some(32_768),
        };
    }

    if model == "k2p5" || model.contains("kimi-k2.5") || model.contains("kimi-k2-thinking") {
        return ModelContextLimits {
            context_window: Some(262_144),
            max_output_tokens: Some(32_768),
        };
    }

    ModelContextLimits::default()
}
