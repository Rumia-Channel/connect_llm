use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Anthropic",
        default_base_url: "https://api.anthropic.com",
        default_model: "claude-sonnet-4-20250514",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Anthropic,
        api_style: ApiStyle::Anthropic,
    }
}
