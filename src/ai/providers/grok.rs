use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Grok",
        default_base_url: "https://api.x.ai/v1",
        default_model: "grok-4",
        supports_thinking_output: false,
        supports_thinking_config: false,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Grok,
        api_style: ApiStyle::OpenAi,
    }
}
