pub mod anthropic;
pub mod gemini;
pub mod github_copilot;
pub mod google_ai_studio;
pub mod kimi;
pub mod kimi_coding;
pub mod openai;
pub mod openai_codex;
pub mod sakura;
pub mod zai;
pub mod zai_coding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Generic,
    OpenAiReasoning,
    Claude,
    Gemini,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPolicyProfile {
    Anthropic,
    Gemini,
    GitHubCopilot,
    OpenAi,
    OpenAiCodex,
    Sakura,
    Kimi,
    ZAi,
    GoogleAiStudio,
}

#[derive(Debug, Clone, Copy)]
pub struct RequestPolicy {
    pub allow_temperature: bool,
    pub allow_reasoning_effort: bool,
    pub allow_thinking_budget: bool,
}

impl RequestPolicy {
    pub fn sanitize_temperature(&self, temperature: Option<f32>) -> Option<f32> {
        if self.allow_temperature {
            temperature
        } else {
            None
        }
    }

    pub fn sanitize_reasoning_effort(
        &self,
        reasoning_effort: Option<&'static str>,
    ) -> Option<&'static str> {
        if self.allow_reasoning_effort {
            reasoning_effort
        } else {
            None
        }
    }

    pub fn sanitize_thinking_budget(&self, thinking_budget: Option<u32>) -> Option<u32> {
        if self.allow_thinking_budget {
            thinking_budget
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStyle {
    Anthropic,
    Gemini,
    GitHubCopilot,
    OpenAi,
    OpenAiCodex,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderSpec {
    pub name: &'static str,
    pub default_base_url: &'static str,
    pub default_model: &'static str,
    pub supports_thinking_output: bool,
    pub supports_thinking_config: bool,
    pub supports_tools: bool,
    pub request_policy_profile: RequestPolicyProfile,
    pub api_style: ApiStyle,
}

impl ProviderSpec {
    pub fn request_policy(&self, model: &str) -> RequestPolicy {
        let model_family = classify_model_family(model);

        match self.request_policy_profile {
            RequestPolicyProfile::Anthropic
            | RequestPolicyProfile::Gemini
            | RequestPolicyProfile::Kimi
            | RequestPolicyProfile::ZAi
            | RequestPolicyProfile::Sakura
            | RequestPolicyProfile::GoogleAiStudio => RequestPolicy {
                allow_temperature: true,
                allow_reasoning_effort: false,
                allow_thinking_budget: false,
            },
            RequestPolicyProfile::OpenAi => RequestPolicy {
                allow_temperature: !matches!(model_family, ModelFamily::OpenAiReasoning),
                allow_reasoning_effort: false,
                allow_thinking_budget: false,
            },
            RequestPolicyProfile::OpenAiCodex => RequestPolicy {
                allow_temperature: false,
                allow_reasoning_effort: false,
                allow_thinking_budget: false,
            },
            RequestPolicyProfile::GitHubCopilot => match model_family {
                ModelFamily::Claude => RequestPolicy {
                    allow_temperature: true,
                    allow_reasoning_effort: false,
                    allow_thinking_budget: true,
                },
                ModelFamily::Gemini => RequestPolicy {
                    allow_temperature: true,
                    allow_reasoning_effort: false,
                    allow_thinking_budget: false,
                },
                ModelFamily::OpenAiReasoning => RequestPolicy {
                    allow_temperature: false,
                    allow_reasoning_effort: true,
                    allow_thinking_budget: false,
                },
                ModelFamily::Generic => RequestPolicy {
                    allow_temperature: true,
                    allow_reasoning_effort: false,
                    allow_thinking_budget: false,
                },
            },
        }
    }
}

pub fn classify_model_family(model: &str) -> ModelFamily {
    let model = model.trim().to_ascii_lowercase();
    if model.contains("claude") {
        return ModelFamily::Claude;
    }
    if model.contains("gemini") {
        return ModelFamily::Gemini;
    }
    if model.starts_with("gpt-5")
        || model.contains("codex")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("o5")
    {
        return ModelFamily::OpenAiReasoning;
    }
    ModelFamily::Generic
}

pub fn openai_compatible_spec_for_base_url(base_url: &str) -> ProviderSpec {
    let normalized = base_url.trim_end_matches('/').to_ascii_lowercase();
    if normalized.contains("moonshot.ai") {
        kimi::spec()
    } else if normalized.contains("api.z.ai") {
        zai::spec()
    } else if normalized.contains("generativelanguage.googleapis.com") {
        google_ai_studio::spec()
    } else if normalized.contains("api.ai.sakura.ad.jp") {
        sakura::spec()
    } else {
        openai::spec()
    }
}
