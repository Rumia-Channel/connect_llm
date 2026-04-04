pub mod ai;
pub mod context;
pub mod mcp;

pub use ai::{
    AiClient, AiConfig, AiError, AiProvider, ChatRequest, ChatResponse, DebugTrace, GeneratedImage,
    GitHubCopilotDeviceAuth, GitHubCopilotDeviceAuthOptions, Message, OpenAiCodexBrowserAuth,
    OpenAiCodexBrowserAuthOptions, StreamChunk, ThinkingConfig, ThinkingDisplay, ThinkingEffort,
    ThinkingOutput, ToolCall, ToolCallDelta, ToolChoice, ToolDefinition, Usage,
    debug_logging_enabled, github_copilot_auth_path, login_github_copilot_via_device,
    login_openai_codex_via_browser, openai_codex_auth_path, set_debug_logging,
};
pub use context::{
    ContextCompaction, ContextManager, ContextManagerConfig, ManagedChatResponse,
    ModelContextLimits, PreparedChatRequest, TextWindow, TextWindowConfig,
    resolve_model_context_limits, split_text_into_windows,
};
pub use mcp::{
    McpBridge, McpConfig, McpManagedChatResponse, McpServerConfig, McpToolExecution,
    McpToolLoopConfig,
};
