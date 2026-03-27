use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Kimi",
        default_base_url: "https://api.moonshot.ai",
        default_model: "kimi-k2.5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Kimi,
        api_style: ApiStyle::OpenAi,
    }
}
