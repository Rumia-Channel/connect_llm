use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "Gemini",
        default_base_url: "https://generativelanguage.googleapis.com/v1beta",
        default_model: "gemini-2.5-pro",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::Gemini,
        api_style: ApiStyle::Gemini,
    }
}
