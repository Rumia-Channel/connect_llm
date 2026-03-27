use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "OpenAiCodex",
        default_base_url: "https://chatgpt.com/backend-api",
        default_model: "gpt-5.1-codex-max",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::OpenAiCodex,
    }
}
