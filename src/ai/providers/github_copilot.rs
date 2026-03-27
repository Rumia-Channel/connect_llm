use super::{ApiStyle, ProviderSpec};

pub fn spec() -> ProviderSpec {
    ProviderSpec {
        name: "GitHubCopilot",
        default_base_url: "https://api.githubcopilot.com",
        default_model: "gpt-4o",
        supports_thinking_output: true,
        supports_thinking_config: true,
        api_style: ApiStyle::GitHubCopilot,
    }
}
