use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "ZAiCoding",
        default_base_url: "https://api.z.ai/api/coding/paas/v4",
        default_model: "glm-5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::OpenAi,
    }
}
