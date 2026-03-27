use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Sakura",
        default_base_url: "https://api.ai.sakura.ad.jp",
        default_model: "gpt-oss-120b",
        supports_thinking_output: true,
        supports_thinking_config: false,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Sakura,
        api_style: ApiStyle::OpenAi,
    }
}
