use super::{ApiStyle, ProviderSpec, RequestPolicyProfile};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "GitHubCopilot",
        default_base_url: "https://api.githubcopilot.com",
        default_model: "claude-sonnet-4.5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        request_policy_profile: RequestPolicyProfile::GitHubCopilot,
        api_style: ApiStyle::GitHubCopilot,
    }
}
