pub mod ai;
pub mod context;
pub mod mcp;

pub use ai::{
    AiAuth, AiClient, AiConfig, AiEndpointConfig, AiError, AiErrorKind, AiHttpConfig, AiProvider,
    ChatRequest, ChatResponse, ContentPart, DebugTrace, GeneratedImage, GitHubCopilotDeviceAuth,
    GitHubCopilotDeviceAuthOptions, ImageDetail, ImageSource, InputImage, Message,
    MultimodalChatRequest, OpenAiCodexBrowserAuth, OpenAiCodexBrowserAuthOptions, RequestMessage,
    StreamChunk, ThinkingConfig, ThinkingDisplay, ThinkingEffort, ThinkingOutput, ToolCall,
    ToolCallDelta, ToolChoice, ToolDefinition, Usage, debug_logging_enabled,
    github_copilot_auth_path, login_github_copilot_via_device, login_openai_codex_via_browser,
    openai_codex_auth_path, set_debug_logging,
};
pub use context::{
    ContextCompaction, ContextManager, ContextManagerConfig, ManagedChatResponse,
    ModelContextLimits, PreparedChatRequest, TextWindow, TextWindowConfig,
    resolve_model_context_limits, split_text_into_windows,
};
pub use mcp::{
    McpBridge, McpConfig, McpConfiguredServerStatus, McpExportedToolStatus, McpManagedChatResponse,
    McpRuntime, McpRuntimeStatus, McpServerConfig, McpStreamEvent, McpToolExecution,
    McpToolLoopConfig,
};
