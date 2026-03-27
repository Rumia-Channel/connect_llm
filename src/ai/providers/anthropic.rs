use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Anthropic",
        default_base_url: "https://api.anthropic.com",
        default_model: "claude-3-5-sonnet-20241022",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::Anthropic,
    }
}
