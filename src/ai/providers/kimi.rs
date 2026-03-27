use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Kimi",
        default_base_url: "https://api.moonshot.ai",
        default_model: "kimi-k2.5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::OpenAi,
    }
}
