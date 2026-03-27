use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "KimiCoding",
        default_base_url: "https://api.kimi.com/coding",
        default_model: "k2p5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Anthropic,
        api_style: ApiStyle::Anthropic,
    }
}
