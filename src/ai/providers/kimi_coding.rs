use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "KimiCoding",
        default_base_url: "https://api.kimi.com/coding",
        default_model: "kimi-for-coding",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        api_style: ApiStyle::Anthropic,
    }
}
