use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "GoogleAiStudio",
        default_base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        default_model: "gemini-2.5-flash",
        supports_thinking_output: false,
        supports_thinking_config: true,
        api_style: ApiStyle::OpenAi,
    }
}
