pub mod anthropic;
pub mod kimi;
pub mod kimi_coding;
pub mod openai;
pub mod sakura;
pub mod zai;
pub mod zai_coding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStyle {
    Anthropic,
    OpenAi,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderSpec {
    pub name: &'static str,
    pub default_base_url: &'static str,
    pub default_model: &'static str,
    pub supports_thinking_output: bool,
    pub supports_thinking_config: bool,
    pub api_style: ApiStyle,
}
