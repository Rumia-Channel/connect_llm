use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "ZAi",
        default_base_url: "https://api.z.ai/api/paas/v4",
        default_model: "glm-5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::ZAi,
        api_style: ApiStyle::OpenAi,
    }
}
