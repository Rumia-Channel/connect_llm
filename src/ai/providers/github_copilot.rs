use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "GitHubCopilot",
        default_base_url: "https://api.githubcopilot.com",
        default_model: "claude-sonnet-4.5",
        supports_thinking_output: true,
        supports_thinking_config: true,
        supports_tools: true,
        api_style: ApiStyle::GitHubCopilot,
    }
}
