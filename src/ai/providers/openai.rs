use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "OpenAi",
        default_base_url: "https://api.openai.com",
        default_model: "gpt-5.4",
        supports_thinking_output: false,
        supports_thinking_config: false,
        supports_tools: true,
        api_style: ApiStyle::OpenAi,
    }
}
