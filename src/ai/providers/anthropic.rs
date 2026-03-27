use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Anthropic",
        default_base_url: "https://api.anthropic.com",
        default_model: "claude-sonnet-4-20250514",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::Anthropic,
    }
}
