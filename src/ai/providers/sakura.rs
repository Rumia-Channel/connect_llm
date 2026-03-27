use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Sakura",
        default_base_url: "https://api.ai.sakura.ad.jp",
        default_model: "gpt-oss-120b",
        supports_thinking_output: true,
        supports_thinking_config: false,
        api_style: ApiStyle::OpenAi,
    }
}
