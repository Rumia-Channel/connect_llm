use crate::{
    ai::{
        AiClient, AiError, ChatRequest, ChatResponse, DebugTrace, GeneratedImage, Message,
        StreamChunk, ThinkingOutput, ToolCall, ToolCallDelta, ToolDefinition, Usage,
        debug_logging_enabled,
    },
    context::{ContextCompaction, ContextManager, ManagedChatResponse, PreparedChatRequest},
};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::{StreamExt, stream, stream::BoxStream};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, Content, Tool},
    service::{Peer, RunningService},
    transport::{
        TokioChildProcess,
        auth::{
            AuthError, AuthorizationManager, AuthorizationMetadata, CredentialStore,
            OAuthClientConfig, StoredCredentials,
        },
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{ErrorKind, Read, Write},
    net::TcpListener,
    path::Path,
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};
use tokio::process::Command;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

const MAX_IDENTICAL_TOOL_CALLS: usize = 2;
const MCP_PROTOCOL_VERSION_STREAMABLE_HTTP: &str = "2025-11-25";
const MCP_PROTOCOL_VERSION_LEGACY_SSE: &str = "2024-11-05";
const DEFAULT_OAUTH_TIMEOUT_SECS: u64 = 300;
const SUPPORTED_STREAMABLE_HTTP_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26"];
const SUPPORTED_LEGACY_SSE_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers", alias = "servers")]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
}

impl McpConfig {
    pub fn from_json_str(json: &str) -> Result<Self, AiError> {
        serde_json::from_str(json).map_err(|error| AiError::parse(error.to_string()))
    }

    pub fn from_json_value(value: Value) -> Result<Self, AiError> {
        serde_json::from_value(value).map_err(|error| AiError::parse(error.to_string()))
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, AiError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| AiError::api(error.to_string()))?;
        let mut config = Self::from_json_str(&text)?;
        config.expand_env_vars()?;
        if let Some(base_dir) = path.parent() {
            config.resolve_relative_paths(base_dir);
        }
        Ok(config)
    }

    pub fn is_empty(&self) -> bool {
        self.mcp_servers.values().all(|server| !server.enabled) || self.mcp_servers.is_empty()
    }

    fn resolve_relative_paths(&mut self, base_dir: &Path) {
        for server in self.mcp_servers.values_mut() {
            if let Some(command) = &server.command {
                let command_path = PathBuf::from(command);
                if command_path.is_relative() && looks_like_filesystem_path(command) {
                    server.command =
                        Some(base_dir.join(command_path).to_string_lossy().into_owned());
                }
            }
            if let Some(cwd) = &server.cwd {
                if cwd.is_relative() {
                    server.cwd = Some(base_dir.join(cwd));
                }
            }
            if let Some(headers_helper) = &server.headers_helper {
                let helper_path = PathBuf::from(headers_helper);
                if helper_path.is_relative() && looks_like_filesystem_path(headers_helper) {
                    server.headers_helper =
                        Some(base_dir.join(helper_path).to_string_lossy().into_owned());
                }
            }
        }
    }

    fn expand_env_vars(&mut self) -> Result<(), AiError> {
        for (label, server) in &mut self.mcp_servers {
            server.expand_env_vars(label)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default, rename = "type")]
    pub server_type: Option<String>,
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub cwd: Option<PathBuf>,
    #[serde(default, alias = "serverUrl", alias = "uri")]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub headers_helper: Option<String>,
    #[serde(default)]
    pub oauth: Option<McpOAuthConfig>,
    pub description: Option<String>,
    pub transport: Option<String>,
    #[serde(default, alias = "authorization")]
    pub auth_header: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpOAuthConfig {
    pub client_id: Option<String>,
    pub callback_port: Option<u16>,
    pub auth_server_metadata_url: Option<String>,
    pub xaa: Option<bool>,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            server_type: None,
            command: None,
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            url: None,
            headers: BTreeMap::new(),
            headers_helper: None,
            oauth: None,
            description: None,
            transport: None,
            auth_header: None,
        }
    }
}

impl McpServerConfig {
    fn expand_env_vars(&mut self, server_label: &str) -> Result<(), AiError> {
        if let Some(command) = &mut self.command {
            *command = expand_env_vars_in_string(command, server_label, "command")?;
        }
        for argument in &mut self.args {
            *argument = expand_env_vars_in_string(argument, server_label, "args")?;
        }
        for (key, value) in &mut self.env {
            *value = expand_env_vars_in_string(value, server_label, &format!("env.{}", key))?;
        }
        if let Some(cwd) = &mut self.cwd {
            *cwd = PathBuf::from(expand_env_vars_in_string(
                &cwd.to_string_lossy(),
                server_label,
                "cwd",
            )?);
        }
        if let Some(url) = &mut self.url {
            *url = expand_env_vars_in_string(url, server_label, "url")?;
        }
        for (key, value) in &mut self.headers {
            *value = expand_env_vars_in_string(value, server_label, &format!("headers.{}", key))?;
        }
        if let Some(headers_helper) = &mut self.headers_helper {
            *headers_helper =
                expand_env_vars_in_string(headers_helper, server_label, "headersHelper")?;
        }
        if let Some(auth_header) = &mut self.auth_header {
            *auth_header = expand_env_vars_in_string(auth_header, server_label, "authHeader")?;
        }
        if let Some(oauth) = &mut self.oauth {
            if let Some(client_id) = &mut oauth.client_id {
                *client_id = expand_env_vars_in_string(client_id, server_label, "oauth.clientId")?;
            }
            if let Some(metadata_url) = &mut oauth.auth_server_metadata_url {
                *metadata_url = expand_env_vars_in_string(
                    metadata_url,
                    server_label,
                    "oauth.authServerMetadataUrl",
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpToolLoopConfig {
    pub max_round_trips: usize,
}

impl Default for McpToolLoopConfig {
    fn default() -> Self {
        Self { max_round_trips: 8 }
    }
}

#[derive(Debug, Clone)]
pub struct McpManagedChatResponse {
    pub response: ChatResponse,
    pub compaction: Option<ContextCompaction>,
    pub messages: Vec<Message>,
    pub tool_executions: Vec<McpToolExecution>,
}

#[derive(Debug, Clone)]
pub enum McpStreamEvent {
    Chunk(StreamChunk),
    Finished(McpManagedChatResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRuntimeStatus {
    pub connected: bool,
    pub configured_server_count: usize,
    pub connected_server_count: usize,
    pub configured_servers: Vec<McpConfiguredServerStatus>,
    pub exported_tools: Vec<McpExportedToolStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfiguredServerStatus {
    pub label: String,
    pub enabled: bool,
    pub connected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub transport: String,
    pub command: Option<String>,
    pub url: Option<String>,
    pub cwd: Option<PathBuf>,
    pub exported_tool_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpExportedToolStatus {
    pub alias: String,
    pub server_label: String,
    pub remote_tool_name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolExecution {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub result: Value,
    pub server_label: String,
    pub remote_tool_name: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub struct McpBridge {
    config: McpConfig,
    tool_loop: McpToolLoopConfig,
}

pub struct McpRuntime {
    bridge: McpBridge,
    sessions: Option<McpSessionSet>,
}

impl McpBridge {
    pub fn new(config: McpConfig) -> Self {
        Self {
            config,
            tool_loop: McpToolLoopConfig::default(),
        }
    }

    pub fn with_tool_loop_config(mut self, tool_loop: McpToolLoopConfig) -> Self {
        self.tool_loop = tool_loop;
        self
    }

    pub fn config(&self) -> &McpConfig {
        &self.config
    }

    pub fn tool_loop_config(&self) -> McpToolLoopConfig {
        self.tool_loop
    }

    pub async fn connect(&self) -> Result<McpRuntime, AiError> {
        McpRuntime::connect_with_bridge(self.clone()).await
    }

    pub fn status(&self) -> McpRuntimeStatus {
        McpRuntimeStatus::from_config(&self.config)
    }

    pub fn chat_stream<'a>(
        &'a self,
        client: &'a dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        self.chat_stream_inner(None, client, request)
    }

    pub fn chat_stream_with_context_manager<'a>(
        &'a self,
        context_manager: &'a ContextManager,
        client: &'a dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        self.chat_stream_inner(Some(context_manager), client, request)
    }

    pub async fn chat(
        &self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<McpManagedChatResponse, AiError> {
        self.chat_inner(None, client, request).await
    }

    pub async fn chat_with_context_manager(
        &self,
        context_manager: &ContextManager,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<McpManagedChatResponse, AiError> {
        self.chat_inner(Some(context_manager), client, request)
            .await
    }

    async fn chat_inner(
        &self,
        context_manager: Option<&ContextManager>,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<McpManagedChatResponse, AiError> {
        let mut sessions = McpSessionSet::connect(&self.config).await?;
        let result = self
            .chat_with_sessions(context_manager, client, request, &mut sessions)
            .await;
        sessions.close().await;
        result
    }

    fn chat_stream_inner<'a>(
        &'a self,
        context_manager: Option<&'a ContextManager>,
        client: &'a dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        stream! {
            let mut sessions = match McpSessionSet::connect(&self.config).await {
                Ok(sessions) => sessions,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            {
                let mut bridge_stream = self.chat_stream_with_sessions(
                    context_manager,
                    client,
                    request,
                    &mut sessions,
                );
                while let Some(event) = bridge_stream.next().await {
                    yield event;
                }
            }
            sessions.close().await;
        }
        .boxed()
    }

    fn chat_stream_with_sessions<'a>(
        &'a self,
        context_manager: Option<&'a ContextManager>,
        client: &'a dyn AiClient,
        request: ChatRequest,
        sessions: &'a mut McpSessionSet,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        stream! {
            if let Err(error) = sessions.refresh_remote_tools_if_needed().await {
                yield Err(error);
                return;
            }
            let mut request = request;
            let base_tools = request.tools.clone();
            request.tools = base_tools.clone();
            request.tools.extend(sessions.tool_definitions());

            let mut tool_executions = Vec::new();
            let mut tool_call_counts = HashMap::<String, usize>::new();
            let mut pending_error = None;
            let mut finished = false;

            'rounds: for round in 0..=self.tool_loop.max_round_trips {
                if let Err(error) = sessions.refresh_remote_tools_if_needed().await {
                    pending_error = Some(error);
                    break 'rounds;
                }
                request.tools = base_tools.clone();
                request.tools.extend(sessions.tool_definitions());
                let prepared = match prepare_stream_request(context_manager, client, request.clone()).await {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        pending_error = Some(error);
                        break 'rounds;
                    }
                };
                let mut provider_stream = client.chat_stream(prepared.request.clone());
                let mut response_builder = StreamResponseBuilder::new(prepared.request.model.clone());
                let mut buffered_chunks = Vec::new();

                while let Some(next) = provider_stream.next().await {
                    match next {
                        Ok(chunk) => {
                            response_builder.ingest(&chunk);
                            let done = chunk.done;
                            buffered_chunks.push(chunk);
                            if done {
                                break;
                            }
                        }
                        Err(error) => {
                            pending_error = Some(error);
                            break 'rounds;
                        }
                    }
                }

                let response = response_builder.finish();
                let tool_calls = response.tool_calls.clone();
                request.messages.push(assistant_message_from_response(&response));

                if tool_calls.is_empty() {
                    if needs_final_visible_answer(&response, &request) {
                        let followup_request = build_final_no_tools_request(
                            request.clone(),
                            "You have already gathered tool results. Do not call more tools. Provide a direct user-visible answer based on the information already gathered.",
                        );
                        let final_prepared = match prepare_stream_request(context_manager, client, followup_request).await {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                pending_error = Some(error);
                                break 'rounds;
                            }
                        };
                        let mut final_stream = client.chat_stream(final_prepared.request.clone());
                        let mut final_builder = StreamResponseBuilder::new(final_prepared.request.model.clone());
                        let mut final_chunks = Vec::new();
                        while let Some(next) = final_stream.next().await {
                            match next {
                                Ok(chunk) => {
                                    final_builder.ingest(&chunk);
                                    let done = chunk.done;
                                    final_chunks.push(chunk);
                                    if done {
                                        break;
                                    }
                                }
                                Err(error) => {
                                    pending_error = Some(error);
                                    break 'rounds;
                                }
                            }
                        }
                        let final_response = final_builder.finish();
                        if final_response.tool_calls.is_empty() {
                            for chunk in final_chunks {
                                yield Ok(McpStreamEvent::Chunk(chunk));
                            }
                        }
                        yield Ok(McpStreamEvent::Finished(McpManagedChatResponse {
                            response: final_response.clone(),
                            compaction: final_prepared.compaction.or(prepared.compaction.clone()),
                            messages: final_managed_messages_from_request(
                                final_response,
                                &request,
                            ),
                            tool_executions,
                        }));
                        finished = true;
                        break 'rounds;
                    }

                    for chunk in buffered_chunks {
                        yield Ok(McpStreamEvent::Chunk(chunk));
                    }
                    yield Ok(McpStreamEvent::Finished(McpManagedChatResponse {
                        response,
                        compaction: prepared.compaction.clone(),
                        messages: request.messages,
                        tool_executions,
                    }));
                    finished = true;
                    break 'rounds;
                }

                let (allowed_calls, blocked_results) =
                    partition_tool_calls_for_execution(&tool_calls, &mut tool_call_counts);
                request.messages.extend(blocked_results);

                if round == self.tool_loop.max_round_trips || allowed_calls.is_empty() {
                    if round == self.tool_loop.max_round_trips {
                        request.messages.extend(tool_limit_block_messages(&allowed_calls));
                    }
                    let final_reason = if round == self.tool_loop.max_round_trips {
                        "MCP tool loop limit reached. Do not call more tools. Answer using the information already gathered and explain any limitations."
                    } else {
                        "Repeated identical MCP tool calls were blocked. Do not call more tools. Answer using the information already gathered and explain any limitations."
                    };
                    let final_prepared = match prepare_stream_request(
                        context_manager,
                        client,
                        build_final_no_tools_request(request.clone(), final_reason),
                    )
                    .await
                    {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            pending_error = Some(error);
                            break 'rounds;
                        }
                    };
                    let mut final_stream = client.chat_stream(final_prepared.request.clone());
                    let mut final_builder = StreamResponseBuilder::new(final_prepared.request.model.clone());
                    let mut final_chunks = Vec::new();
                    while let Some(next) = final_stream.next().await {
                        match next {
                            Ok(chunk) => {
                                final_builder.ingest(&chunk);
                                let done = chunk.done;
                                final_chunks.push(chunk);
                                if done {
                                    break;
                                }
                            }
                            Err(error) => {
                                pending_error = Some(error);
                                break 'rounds;
                            }
                        }
                    }
                    let final_response = final_builder.finish();
                    if final_response.tool_calls.is_empty() {
                        for chunk in final_chunks {
                            yield Ok(McpStreamEvent::Chunk(chunk));
                        }
                    }
                    yield Ok(McpStreamEvent::Finished(McpManagedChatResponse {
                        response: final_response.clone(),
                        compaction: final_prepared.compaction,
                        messages: final_managed_messages_from_request(
                            final_response,
                            &request,
                        ),
                        tool_executions,
                    }));
                    finished = true;
                    break 'rounds;
                }

                let (tool_results, executed) = match sessions.execute_tool_calls(&allowed_calls).await {
                    Ok(result) => result,
                    Err(error) => {
                        pending_error = Some(error);
                        break 'rounds;
                    }
                };
                request.messages.extend(tool_results);
                tool_executions.extend(executed);
            }

            if let Some(error) = pending_error {
                yield Err(error);
                return;
            }

            if !finished {
                yield Err(AiError::api(
                    "MCP streaming tool loop terminated unexpectedly".to_string(),
                ));
            }
        }
        .boxed()
    }

    async fn chat_with_sessions(
        &self,
        context_manager: Option<&ContextManager>,
        client: &dyn AiClient,
        mut request: ChatRequest,
        sessions: &mut McpSessionSet,
    ) -> Result<McpManagedChatResponse, AiError> {
        sessions.refresh_remote_tools_if_needed().await?;
        let base_tools = request.tools.clone();
        request.tools = base_tools.clone();
        request.tools.extend(sessions.tool_definitions());

        let mut tool_executions = Vec::new();
        let mut tool_call_counts = HashMap::<String, usize>::new();

        for round in 0..=self.tool_loop.max_round_trips {
            sessions.refresh_remote_tools_if_needed().await?;
            request.tools = base_tools.clone();
            request.tools.extend(sessions.tool_definitions());
            let managed = send_chat(context_manager, client, request.clone()).await?;
            let compaction = managed.compaction.clone();

            let response = managed.response;
            let assistant_message = assistant_message_from_response(&response);
            let tool_calls = response.tool_calls.clone();
            request.messages.push(assistant_message);

            if tool_calls.is_empty() {
                if needs_final_visible_answer(&response, &request) {
                    let mut final_managed = finalize_chat_without_tools(
                        context_manager,
                        client,
                        request.clone(),
                        "You have already gathered tool results. Do not call more tools. Provide a direct user-visible answer based on the information already gathered.",
                    )
                    .await?;
                    final_managed.compaction = final_managed.compaction.or(compaction);
                    return Ok(McpManagedChatResponse {
                        response: final_managed.response.clone(),
                        compaction: final_managed.compaction,
                        messages: final_managed_messages_from_request(
                            final_managed.response,
                            &request,
                        ),
                        tool_executions,
                    });
                }

                return Ok(McpManagedChatResponse {
                    response,
                    compaction,
                    messages: request.messages,
                    tool_executions,
                });
            }

            let (allowed_calls, blocked_results) =
                partition_tool_calls_for_execution(&tool_calls, &mut tool_call_counts);
            request.messages.extend(blocked_results);

            if round == self.tool_loop.max_round_trips || allowed_calls.is_empty() {
                if round == self.tool_loop.max_round_trips {
                    request
                        .messages
                        .extend(tool_limit_block_messages(&allowed_calls));
                }
                let final_reason = if round == self.tool_loop.max_round_trips {
                    "MCP tool loop limit reached. Do not call more tools. Answer using the information already gathered and explain any limitations."
                } else {
                    "Repeated identical MCP tool calls were blocked. Do not call more tools. Answer using the information already gathered and explain any limitations."
                };
                let mut final_managed = finalize_chat_without_tools(
                    context_manager,
                    client,
                    request.clone(),
                    final_reason,
                )
                .await?;
                final_managed.compaction = final_managed.compaction.or(compaction);
                return Ok(McpManagedChatResponse {
                    response: final_managed.response.clone(),
                    compaction: final_managed.compaction,
                    messages: final_managed_messages_from_request(final_managed.response, &request),
                    tool_executions,
                });
            }

            let (tool_results, executed) = sessions.execute_tool_calls(&allowed_calls).await?;
            request.messages.extend(tool_results);
            tool_executions.extend(executed);
        }

        let final_managed = finalize_chat_without_tools(
            context_manager,
            client,
            request.clone(),
            "MCP tool loop terminated unexpectedly. Do not call more tools. Answer using the information already gathered and explain any limitations.",
        )
        .await?;
        Ok(McpManagedChatResponse {
            response: final_managed.response.clone(),
            compaction: final_managed.compaction,
            messages: final_managed_messages_from_request(final_managed.response, &request),
            tool_executions,
        })
    }
}

impl McpRuntime {
    pub async fn connect(config: McpConfig) -> Result<Self, AiError> {
        McpBridge::new(config).connect().await
    }

    async fn connect_with_bridge(bridge: McpBridge) -> Result<Self, AiError> {
        let sessions = McpSessionSet::connect(bridge.config()).await?;
        Ok(Self {
            bridge,
            sessions: Some(sessions),
        })
    }

    pub fn bridge(&self) -> &McpBridge {
        &self.bridge
    }

    pub fn status(&self) -> McpRuntimeStatus {
        match &self.sessions {
            Some(sessions) => sessions.status_with_config(self.bridge.config()),
            None => self.bridge.status(),
        }
    }

    pub async fn close(&mut self) {
        if let Some(sessions) = self.sessions.take() {
            sessions.close().await;
        }
    }

    pub async fn chat(
        &mut self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<McpManagedChatResponse, AiError> {
        match (&self.bridge, self.sessions.as_mut()) {
            (bridge, Some(sessions)) => {
                bridge
                    .chat_with_sessions(None, client, request, sessions)
                    .await
            }
            (_, None) => Err(AiError::api(
                "MCP runtime is closed. Reconnect before sending requests.".to_string(),
            )),
        }
    }

    pub async fn chat_with_context_manager(
        &mut self,
        context_manager: &ContextManager,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<McpManagedChatResponse, AiError> {
        match (&self.bridge, self.sessions.as_mut()) {
            (bridge, Some(sessions)) => {
                bridge
                    .chat_with_sessions(Some(context_manager), client, request, sessions)
                    .await
            }
            (_, None) => Err(AiError::api(
                "MCP runtime is closed. Reconnect before sending requests.".to_string(),
            )),
        }
    }

    pub fn chat_stream<'a>(
        &'a mut self,
        client: &'a dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        match (&self.bridge, self.sessions.as_mut()) {
            (bridge, Some(sessions)) => {
                bridge.chat_stream_with_sessions(None, client, request, sessions)
            }
            (_, None) => stream::once(async {
                Err(AiError::api(
                    "MCP runtime is closed. Reconnect before sending requests.".to_string(),
                ))
            })
            .boxed(),
        }
    }

    pub fn chat_stream_with_context_manager<'a>(
        &'a mut self,
        context_manager: &'a ContextManager,
        client: &'a dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<McpStreamEvent, AiError>> {
        match (&self.bridge, self.sessions.as_mut()) {
            (bridge, Some(sessions)) => {
                bridge.chat_stream_with_sessions(Some(context_manager), client, request, sessions)
            }
            (_, None) => stream::once(async {
                Err(AiError::api(
                    "MCP runtime is closed. Reconnect before sending requests.".to_string(),
                ))
            })
            .boxed(),
        }
    }
}

struct PendingMcpServer {
    server_label: String,
    description: Option<String>,
    session: PendingMcpSession,
    tools: Vec<Tool>,
}

enum PendingMcpSession {
    Local {
        service: RunningService<RoleClient, ()>,
        peer: Peer<RoleClient>,
    },
    Remote(RemoteMcpClient),
}

impl PendingMcpServer {
    async fn connect(server_label: &str, config: &McpServerConfig) -> Result<Self, AiError> {
        match detect_transport(config)? {
            McpTransport::Stdio => {
                let service = connect_stdio_peer(config).await?;
                Self::from_service(
                    server_label.to_string(),
                    config.description.clone(),
                    service,
                )
                .await
            }
            McpTransport::StreamableHttp | McpTransport::LegacySse => {
                let client = RemoteMcpClient::connect(server_label, config).await?;
                let tools = client.list_all_tools().await?;
                Ok(Self {
                    server_label: server_label.to_string(),
                    description: config.description.clone(),
                    session: PendingMcpSession::Remote(client),
                    tools,
                })
            }
        }
    }

    async fn from_service(
        server_label: String,
        description: Option<String>,
        service: RunningService<RoleClient, ()>,
    ) -> Result<Self, AiError> {
        let peer = service.peer().clone();
        let tools = peer.list_all_tools().await.map_err(mcp_service_error)?;
        Ok(Self {
            server_label,
            description,
            session: PendingMcpSession::Local { service, peer },
            tools,
        })
    }
}

struct McpSessionSet {
    servers: Vec<McpServerSession>,
    exported_tools: Vec<ToolDefinition>,
    tool_index: HashMap<String, ResolvedMcpTool>,
    connect_errors: HashMap<String, String>,
}

impl McpSessionSet {
    async fn connect(config: &McpConfig) -> Result<Self, AiError> {
        if config.is_empty() {
            return Ok(Self {
                servers: Vec::new(),
                exported_tools: Vec::new(),
                tool_index: HashMap::new(),
                connect_errors: HashMap::new(),
            });
        }

        let mut pending = Vec::new();
        let mut connect_errors = HashMap::new();
        for (label, server) in &config.mcp_servers {
            if !server.enabled {
                continue;
            }
            match PendingMcpServer::connect(label, server).await {
                Ok(connected) => pending.push(connected),
                Err(error) => {
                    connect_errors.insert(label.clone(), error.to_string());
                }
            }
        }

        Ok(Self::from_pending(pending, connect_errors))
    }

    fn from_pending(
        pending: Vec<PendingMcpServer>,
        connect_errors: HashMap<String, String>,
    ) -> Self {
        let mut servers = Vec::new();

        for pending_server in pending {
            servers.push(McpServerSession {
                server_label: pending_server.server_label,
                description: pending_server.description,
                session: pending_server.session,
                tools: pending_server.tools,
                aliases: HashMap::new(),
            });
        }

        let mut this = Self {
            servers,
            exported_tools: Vec::new(),
            tool_index: HashMap::new(),
            connect_errors,
        };
        this.rebuild_tool_catalog();
        this
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.exported_tools.clone()
    }

    async fn refresh_remote_tools_if_needed(&mut self) -> Result<(), AiError> {
        let mut changed = false;
        for server in &mut self.servers {
            if server.refresh_tools_if_needed().await? {
                changed = true;
            }
        }
        if changed {
            self.rebuild_tool_catalog();
        }
        Ok(())
    }

    fn status_with_config(&self, config: &McpConfig) -> McpRuntimeStatus {
        let connected_labels = self
            .servers
            .iter()
            .map(|server| server.server_label.as_str())
            .collect::<HashSet<_>>();

        let mut tool_counts = HashMap::<String, usize>::new();
        for tool in self.exported_tools_status() {
            *tool_counts.entry(tool.server_label).or_default() += 1;
        }

        let configured_servers = config
            .mcp_servers
            .iter()
            .map(|(label, server)| McpConfiguredServerStatus {
                label: label.clone(),
                enabled: server.enabled,
                connected: server.enabled && connected_labels.contains(label.as_str()),
                last_error: self.connect_errors.get(label).cloned(),
                transport: transport_label(server),
                command: server.command.clone(),
                url: server.url.clone(),
                cwd: server.cwd.clone(),
                exported_tool_count: tool_counts.get(label).copied().unwrap_or_default(),
            })
            .collect::<Vec<_>>();

        McpRuntimeStatus {
            connected: !self.servers.is_empty(),
            configured_server_count: configured_servers.len(),
            connected_server_count: self.servers.len(),
            configured_servers,
            exported_tools: self.exported_tools_status(),
        }
    }

    fn exported_tools_status(&self) -> Vec<McpExportedToolStatus> {
        self.exported_tools
            .iter()
            .filter_map(|tool| {
                let resolved = self.tool_index.get(&tool.name)?;
                let server = self.servers.get(resolved.server_index)?;
                let remote_tool_name = server.aliases.get(&tool.name)?.clone();
                Some(McpExportedToolStatus {
                    alias: tool.name.clone(),
                    server_label: resolved.server_label.clone(),
                    remote_tool_name,
                    description: tool.description.clone(),
                })
            })
            .collect()
    }

    async fn close(mut self) {
        for server in &mut self.servers {
            let _ = server.close().await;
        }
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
    ) -> Result<(Vec<Message>, Vec<McpToolExecution>), AiError> {
        let mut messages = Vec::with_capacity(tool_calls.len());
        let mut executions = Vec::with_capacity(tool_calls.len());

        for tool_call in tool_calls {
            let resolved = self.tool_index.get(&tool_call.name).ok_or_else(|| {
                AiError::api(format!(
                    "MCP bridge cannot execute unknown tool call '{}'",
                    tool_call.name
                ))
            })?;
            let server = self.servers.get(resolved.server_index).ok_or_else(|| {
                AiError::api(format!(
                    "MCP bridge lost session for server '{}'",
                    resolved.server_label
                ))
            })?;
            let remote_tool_name =
                server
                    .aliases
                    .get(&tool_call.name)
                    .cloned()
                    .ok_or_else(|| {
                        AiError::api(format!(
                            "MCP bridge cannot map tool '{}' back to server '{}'",
                            tool_call.name, resolved.server_label
                        ))
                    })?;
            let arguments = value_as_json_object(&tool_call.arguments)?;
            let result = server.call_tool(&remote_tool_name, arguments).await?;
            let result_value = call_tool_result_to_value(&result);
            let is_error = result.is_error.unwrap_or(false);
            let mut message = Message::tool_result(
                tool_call.id.clone(),
                tool_call.name.clone(),
                result_value.clone(),
            );
            if is_error {
                message.set_tool_error(true);
            }
            messages.push(message);
            executions.push(McpToolExecution {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
                result: result_value,
                server_label: resolved.server_label.clone(),
                remote_tool_name,
                is_error,
            });
        }

        Ok((messages, executions))
    }

    fn rebuild_tool_catalog(&mut self) {
        let mut used_aliases = HashSet::new();
        let mut exported_tools = Vec::new();
        let mut tool_index = HashMap::new();

        for (server_index, server) in self.servers.iter_mut().enumerate() {
            server.aliases.clear();
            for tool in &server.tools {
                let remote_tool_name = tool.name.to_string();
                let alias =
                    allocate_tool_alias(&server.server_label, &remote_tool_name, &mut used_aliases);
                let input_schema = Value::Object((*tool.input_schema).clone());
                let description = build_tool_description(
                    &server.server_label,
                    server.description.as_deref(),
                    &remote_tool_name,
                    tool.description.as_deref(),
                );

                exported_tools.push(ToolDefinition::function(
                    alias.clone(),
                    Some(description),
                    input_schema,
                ));
                server.aliases.insert(alias.clone(), remote_tool_name);
                tool_index.insert(
                    alias,
                    ResolvedMcpTool {
                        server_index,
                        server_label: server.server_label.clone(),
                    },
                );
            }
        }

        self.exported_tools = exported_tools;
        self.tool_index = tool_index;
    }
}

struct McpServerSession {
    server_label: String,
    description: Option<String>,
    session: PendingMcpSession,
    tools: Vec<Tool>,
    aliases: HashMap<String, String>,
}

impl McpServerSession {
    async fn close(&mut self) -> Result<(), AiError> {
        match &mut self.session {
            PendingMcpSession::Local { service, .. } => {
                service.close().await.map_err(mcp_service_error)?;
            }
            PendingMcpSession::Remote(client) => {
                client.close().await?;
            }
        }
        Ok(())
    }

    async fn refresh_tools_if_needed(&mut self) -> Result<bool, AiError> {
        match &mut self.session {
            PendingMcpSession::Local { .. } => Ok(false),
            PendingMcpSession::Remote(client) => {
                if let Some(tools) = client.refresh_tools_if_needed().await? {
                    self.tools = tools;
                    return Ok(true);
                }
                Ok(false)
            }
        }
    }

    async fn call_tool(
        &self,
        remote_tool_name: &str,
        arguments: Map<String, Value>,
    ) -> Result<rmcp::model::CallToolResult, AiError> {
        match &self.session {
            PendingMcpSession::Local { peer, .. } => peer
                .call_tool(
                    CallToolRequestParams::new(Cow::Owned(remote_tool_name.to_string()))
                        .with_arguments(arguments),
                )
                .await
                .map_err(mcp_service_error),
            PendingMcpSession::Remote(client) => {
                client.call_tool(remote_tool_name, arguments).await
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ResolvedMcpTool {
    server_index: usize,
    server_label: String,
}

fn default_true() -> bool {
    true
}

fn transport_label(config: &McpServerConfig) -> String {
    if let Some(server_type) = config.server_type.as_deref() {
        return server_type.to_string();
    }
    if let Some(transport) = config.transport.as_deref() {
        return transport.to_string();
    }
    if config.command.is_some() && config.url.is_none() {
        return "stdio".to_string();
    }
    if config.url.is_some() && config.command.is_none() {
        return "streamable_http".to_string();
    }
    "unknown".to_string()
}

impl McpRuntimeStatus {
    fn from_config(config: &McpConfig) -> Self {
        let configured_servers = config
            .mcp_servers
            .iter()
            .map(|(label, server)| McpConfiguredServerStatus {
                label: label.clone(),
                enabled: server.enabled,
                connected: false,
                last_error: None,
                transport: transport_label(server),
                command: server.command.clone(),
                url: server.url.clone(),
                cwd: server.cwd.clone(),
                exported_tool_count: 0,
            })
            .collect::<Vec<_>>();

        Self {
            connected: false,
            configured_server_count: configured_servers.len(),
            connected_server_count: 0,
            configured_servers,
            exported_tools: Vec::new(),
        }
    }
}

async fn send_chat(
    context_manager: Option<&ContextManager>,
    client: &dyn AiClient,
    request: ChatRequest,
) -> Result<ManagedChatResponse, AiError> {
    if let Some(context_manager) = context_manager {
        context_manager.chat(client, request).await
    } else {
        let response = client.chat(request).await?;
        Ok(ManagedChatResponse {
            response,
            compaction: None,
        })
    }
}

async fn prepare_stream_request(
    context_manager: Option<&ContextManager>,
    client: &dyn AiClient,
    request: ChatRequest,
) -> Result<PreparedChatRequest, AiError> {
    if let Some(context_manager) = context_manager {
        context_manager
            .prepare_stream_request(client, request)
            .await
    } else {
        Ok(PreparedChatRequest {
            request,
            compaction: None,
        })
    }
}

async fn finalize_chat_without_tools(
    context_manager: Option<&ContextManager>,
    client: &dyn AiClient,
    request: ChatRequest,
    reason: &str,
) -> Result<ManagedChatResponse, AiError> {
    send_chat(
        context_manager,
        client,
        build_final_no_tools_request(request, reason),
    )
    .await
}

fn build_final_no_tools_request(mut request: ChatRequest, reason: &str) -> ChatRequest {
    request.tools.clear();
    request.tool_choice = None;
    request.system = Some(match request.system.take() {
        Some(existing) if !existing.trim().is_empty() => format!("{}\n\n{}", existing, reason),
        _ => reason.to_string(),
    });
    request
}

fn needs_final_visible_answer(response: &ChatResponse, request: &ChatRequest) -> bool {
    response.tool_calls.is_empty()
        && response.content.trim().is_empty()
        && request
            .messages
            .iter()
            .any(|message| message.role() == "tool")
}

fn final_managed_messages_from_request(
    response: ChatResponse,
    request: &ChatRequest,
) -> Vec<Message> {
    let mut messages = request.messages.clone();
    messages.push(assistant_message_from_response(&response));
    messages
}

fn partition_tool_calls_for_execution(
    tool_calls: &[ToolCall],
    counts: &mut HashMap<String, usize>,
) -> (Vec<ToolCall>, Vec<Message>) {
    let mut allowed = Vec::new();
    let mut blocked = Vec::new();

    for tool_call in tool_calls {
        let signature = tool_call_signature(tool_call);
        let next_count = counts.get(&signature).copied().unwrap_or_default() + 1;
        counts.insert(signature, next_count);

        if next_count > MAX_IDENTICAL_TOOL_CALLS {
            let mut message = Message::tool_result(
                tool_call.id.clone(),
                tool_call.name.clone(),
                json!({
                    "error": "repeated_tool_call_blocked",
                    "message": "This exact MCP tool call was already attempted multiple times. Use prior results to answer or try a meaningfully different call.",
                }),
            );
            message.set_tool_error(true);
            blocked.push(message);
        } else {
            allowed.push(tool_call.clone());
        }
    }

    (allowed, blocked)
}

fn tool_limit_block_messages(tool_calls: &[ToolCall]) -> Vec<Message> {
    tool_calls
        .iter()
        .map(|tool_call| {
            let mut message = Message::tool_result(
                tool_call.id.clone(),
                tool_call.name.clone(),
                json!({
                    "error": "tool_loop_limit_reached",
                    "message": "The MCP tool loop limit was reached before this tool call could be executed. Answer using the information already gathered.",
                }),
            );
            message.set_tool_error(true);
            message
        })
        .collect()
}

fn tool_call_signature(tool_call: &ToolCall) -> String {
    format!(
        "{}:{}",
        tool_call.name,
        serde_json::to_string(&tool_call.arguments)
            .unwrap_or_else(|_| tool_call.arguments.to_string())
    )
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponseEnvelope {
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<JsonRpcErrorObject>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcErrorObject {
    code: i64,
    message: String,
    #[serde(default)]
    data: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResultValue {
    protocol_version: String,
}

#[derive(Debug)]
struct RemoteMcpClient {
    inner: RemoteTransport,
}

#[derive(Debug)]
enum RemoteTransport {
    StreamableHttp(RemoteHttpClient),
    LegacySse(LegacySseClient),
}

#[derive(Debug)]
struct RemoteHttpState {
    session_id: Option<String>,
    protocol_version: String,
    next_request_id: u64,
}

#[derive(Debug, Default)]
struct RemoteToolCatalogState {
    tools_dirty: AtomicBool,
    last_event_id: Mutex<Option<String>>,
}

#[derive(Debug)]
struct RemoteHttpListener {
    shutdown_tx: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

#[derive(Debug)]
struct RemoteHttpClient {
    server_label: String,
    url: String,
    client: reqwest::Client,
    oauth: Option<Arc<RemoteOAuthContext>>,
    state: Arc<Mutex<RemoteHttpState>>,
    tool_state: Arc<RemoteToolCatalogState>,
    listener: Arc<Mutex<Option<RemoteHttpListener>>>,
    reinitialize_lock: Arc<Mutex<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RemoteProtocolFlavor {
    StreamableHttp,
    LegacySse,
}

#[derive(Debug)]
struct LegacySseClient {
    base_url: String,
    client: reqwest::Client,
    oauth: Option<Arc<RemoteOAuthContext>>,
    endpoint_url: Mutex<String>,
    protocol_version: Mutex<String>,
    next_request_id: AtomicU64,
    tool_state: Arc<RemoteToolCatalogState>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    shutdown_tx: Mutex<Option<oneshot::Sender<()>>>,
    reader_task: Mutex<Option<JoinHandle<()>>>,
}

struct RemoteOAuthContext {
    server_label: String,
    configured_client_id: Option<String>,
    callback_port: Option<u16>,
    metadata_source: OAuthMetadataSource,
    manager: Mutex<AuthorizationManager>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OAuthMetadataSource {
    Discover,
    Configured,
}

impl std::fmt::Debug for RemoteOAuthContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteOAuthContext")
            .field("server_label", &self.server_label)
            .field("configured_client_id", &self.configured_client_id)
            .field("callback_port", &self.callback_port)
            .field("metadata_source", &self.metadata_source)
            .finish()
    }
}

#[derive(Debug, Clone)]
struct FileCredentialStore {
    path: PathBuf,
}

#[async_trait]
impl CredentialStore for FileCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        if !self.path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.path)
            .map_err(|error| AuthError::InternalError(error.to_string()))?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| AuthError::InternalError(error.to_string()))
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| AuthError::InternalError(error.to_string()))?;
        }
        let text = serde_json::to_string_pretty(&credentials)
            .map_err(|error| AuthError::InternalError(error.to_string()))?;
        fs::write(&self.path, text).map_err(|error| AuthError::InternalError(error.to_string()))
    }

    async fn clear(&self) -> Result<(), AuthError> {
        if self.path.exists() {
            fs::remove_file(&self.path)
                .map_err(|error| AuthError::InternalError(error.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct OAuthCallbackResult {
    code: String,
    state: String,
}

impl RemoteOAuthContext {
    async fn new(
        server_label: &str,
        config: &McpServerConfig,
        client: reqwest::Client,
    ) -> Result<Option<Arc<Self>>, AiError> {
        let Some(oauth_config) = config.oauth.as_ref() else {
            return Ok(None);
        };
        let url = config
            .url
            .as_ref()
            .ok_or_else(|| AiError::parse("Remote MCP server is missing 'url'".to_string()))?;
        let mut manager = AuthorizationManager::new(url.clone())
            .await
            .map_err(map_auth_error)?;
        manager.with_client(client).map_err(map_auth_error)?;
        manager.set_credential_store(FileCredentialStore {
            path: oauth_store_path(server_label, config),
        });

        let metadata_source = match oauth_metadata_source(oauth_config) {
            OAuthMetadataSource::Configured => {
                let metadata_url = oauth_config
                    .auth_server_metadata_url
                    .as_deref()
                    .ok_or_else(|| {
                        AiError::parse(
                            "oauth.authServerMetadataUrl must be set when using configured OAuth metadata"
                                .to_string(),
                        )
                    })?;
                manager.set_metadata(fetch_authorization_metadata(metadata_url).await?);
                OAuthMetadataSource::Configured
            }
            OAuthMetadataSource::Discover => OAuthMetadataSource::Discover,
        };

        let _ = manager
            .initialize_from_store()
            .await
            .map_err(map_auth_error)?;

        Ok(Some(Arc::new(Self {
            server_label: server_label.to_string(),
            configured_client_id: oauth_config.client_id.clone(),
            callback_port: oauth_config.callback_port,
            metadata_source,
            manager: Mutex::new(manager),
        })))
    }

    async fn attach_existing_token(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, AiError> {
        let manager = self.manager.lock().await;
        match manager.get_access_token().await {
            Ok(token) => Ok(request.bearer_auth(token)),
            Err(AuthError::AuthorizationRequired) | Err(AuthError::TokenRefreshFailed(_)) => {
                Ok(request)
            }
            Err(error) => Err(map_auth_error(error)),
        }
    }

    async fn authorize_interactively(&self) -> Result<(), AiError> {
        let mut manager = self.manager.lock().await;
        if self.metadata_source == OAuthMetadataSource::Discover {
            let metadata = manager.discover_metadata().await.map_err(map_auth_error)?;
            manager.set_metadata(metadata);
        }

        let listener =
            TcpListener::bind(("127.0.0.1", self.callback_port.unwrap_or(0))).map_err(|error| {
                AiError::api(format!("Failed to bind OAuth callback port: {}", error))
            })?;
        let port = listener
            .local_addr()
            .map_err(|error| {
                AiError::api(format!("Failed to inspect OAuth callback port: {}", error))
            })?
            .port();
        let redirect_uri = format!("http://127.0.0.1:{}/callback", port);

        let auth_url = if let Some(client_id) = &self.configured_client_id {
            manager
                .configure_client(OAuthClientConfig::new(
                    client_id.clone(),
                    redirect_uri.clone(),
                ))
                .map_err(map_auth_error)?;
            manager
                .get_authorization_url(&[])
                .await
                .map_err(map_auth_error)?
        } else {
            let client_config = manager
                .register_client("connect_llm", &redirect_uri, &[])
                .await
                .map_err(map_auth_error)?;
            manager
                .configure_client(client_config)
                .map_err(map_auth_error)?;
            manager
                .get_authorization_url(&[])
                .await
                .map_err(map_auth_error)?
        };

        open_url_in_browser_local(&auth_url)?;
        let callback =
            wait_for_oauth_callback(listener, Duration::from_secs(DEFAULT_OAUTH_TIMEOUT_SECS))?;
        manager
            .exchange_code_for_token(&callback.code, &callback.state)
            .await
            .map_err(map_auth_error)?;
        Ok(())
    }
}

async fn connect_stdio_peer(
    config: &McpServerConfig,
) -> Result<RunningService<RoleClient, ()>, AiError> {
    validate_supported_config(config)?;
    let command = config
        .command
        .as_ref()
        .ok_or_else(|| AiError::parse("MCP stdio server is missing 'command'".to_string()))?;
    let mut process = Command::new(command);
    process.args(&config.args);
    if let Some(cwd) = &config.cwd {
        process.current_dir(cwd);
    }
    for (key, value) in &config.env {
        process.env(key, value);
    }
    let stderr = if debug_logging_enabled() {
        Stdio::inherit()
    } else {
        Stdio::null()
    };
    let (transport, _captured_stderr) = TokioChildProcess::builder(process)
        .stderr(stderr)
        .spawn()
        .map_err(|error| AiError::api(error.to_string()))?;
    ().serve(transport).await.map_err(mcp_service_error)
}

impl RemoteMcpClient {
    async fn connect(server_label: &str, config: &McpServerConfig) -> Result<Self, AiError> {
        let headers = build_remote_headers(config, Some(server_label)).await?;
        let client = build_remote_reqwest_client(&headers)?;
        let oauth = RemoteOAuthContext::new(server_label, config, client.clone()).await?;

        let inner = match detect_transport(config)? {
            McpTransport::StreamableHttp => RemoteTransport::StreamableHttp(
                RemoteHttpClient::connect(server_label, config, client, oauth).await?,
            ),
            McpTransport::LegacySse => RemoteTransport::LegacySse(
                LegacySseClient::connect(server_label, config, client, oauth).await?,
            ),
            McpTransport::Stdio => {
                return Err(AiError::parse(
                    "RemoteMcpClient cannot be used for stdio transport".to_string(),
                ));
            }
        };

        Ok(Self { inner })
    }

    async fn list_all_tools(&self) -> Result<Vec<Tool>, AiError> {
        match &self.inner {
            RemoteTransport::StreamableHttp(client) => client.list_all_tools().await,
            RemoteTransport::LegacySse(client) => client.list_all_tools().await,
        }
    }

    async fn refresh_tools_if_needed(&self) -> Result<Option<Vec<Tool>>, AiError> {
        match &self.inner {
            RemoteTransport::StreamableHttp(client) => client.refresh_tools_if_needed().await,
            RemoteTransport::LegacySse(client) => client.refresh_tools_if_needed().await,
        }
    }

    async fn call_tool(
        &self,
        remote_tool_name: &str,
        arguments: Map<String, Value>,
    ) -> Result<rmcp::model::CallToolResult, AiError> {
        match &self.inner {
            RemoteTransport::StreamableHttp(client) => {
                client.call_tool(remote_tool_name, arguments).await
            }
            RemoteTransport::LegacySse(client) => {
                client.call_tool(remote_tool_name, arguments).await
            }
        }
    }

    async fn close(&mut self) -> Result<(), AiError> {
        match &mut self.inner {
            RemoteTransport::StreamableHttp(client) => client.close().await,
            RemoteTransport::LegacySse(client) => client.close().await,
        }
    }
}

impl RemoteHttpClient {
    async fn connect(
        server_label: &str,
        config: &McpServerConfig,
        client: reqwest::Client,
        oauth: Option<Arc<RemoteOAuthContext>>,
    ) -> Result<Self, AiError> {
        let url = config
            .url
            .clone()
            .ok_or_else(|| AiError::parse("MCP HTTP server is missing 'url'".to_string()))?;
        let this = Self {
            server_label: server_label.to_string(),
            url,
            client,
            oauth,
            state: Arc::new(Mutex::new(RemoteHttpState {
                session_id: None,
                protocol_version: MCP_PROTOCOL_VERSION_STREAMABLE_HTTP.to_string(),
                next_request_id: 0,
            })),
            tool_state: Arc::new(RemoteToolCatalogState::default()),
            listener: Arc::new(Mutex::new(None)),
            reinitialize_lock: Arc::new(Mutex::new(())),
        };
        this.initialize(server_label).await?;
        Ok(this)
    }

    async fn initialize(&self, server_label: &str) -> Result<(), AiError> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION_STREAMABLE_HTTP,
            "capabilities": {},
            "clientInfo": {
                "name": "connect_llm",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });
        let (value, session_id) = self
            .send_request_internal("initialize", Some(params), false, true)
            .await?
            .ok_or_else(|| AiError::api("MCP initialize did not return a response".to_string()))?;
        let result: InitializeResultValue = serde_json::from_value(value).map_err(|error| {
            AiError::parse(format!("Invalid MCP initialize response: {}", error))
        })?;
        validate_negotiated_protocol_version(
            &result.protocol_version,
            RemoteProtocolFlavor::StreamableHttp,
        )?;
        {
            let mut state = self.state.lock().await;
            state.session_id = session_id;
            state.protocol_version = result.protocol_version;
        }
        let _ = self
            .send_request_internal("notifications/initialized", None, true, false)
            .await?;
        self.start_or_restart_listener(server_label).await;
        Ok(())
    }

    async fn close(&self) -> Result<(), AiError> {
        self.stop_listener().await;
        let (session_id, protocol_version) = {
            let state = self.state.lock().await;
            (state.session_id.clone(), state.protocol_version.clone())
        };
        let Some(session_id) = session_id else {
            return Ok(());
        };

        let mut builder = self.client.delete(&self.url);
        builder = builder
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", protocol_version)
            .header("MCP-Session-Id", session_id);
        if let Some(oauth) = &self.oauth {
            builder = oauth.attach_existing_token(builder).await?;
        }
        let response = builder
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;
        if response.status().is_success()
            || response.status() == reqwest::StatusCode::NO_CONTENT
            || response.status() == reqwest::StatusCode::ACCEPTED
            || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED
            || response.status() == reqwest::StatusCode::NOT_FOUND
        {
            let mut state = self.state.lock().await;
            state.session_id = None;
            return Ok(());
        }
        Err(AiError::api(format!(
            "MCP session delete failed with HTTP {}",
            response.status()
        )))
    }

    async fn list_all_tools(&self) -> Result<Vec<Tool>, AiError> {
        let mut cursor: Option<String> = None;
        let mut tools = Vec::new();
        loop {
            let params = Some(match cursor.as_ref() {
                Some(value) => json!({ "cursor": value }),
                None => json!({}),
            });
            let value = self.send_request_value("tools/list", params).await?;
            let result: rmcp::model::ListToolsResult =
                serde_json::from_value(value).map_err(|error| {
                    AiError::parse(format!("Invalid tools/list response: {}", error))
                })?;
            tools.extend(result.tools);
            cursor = result.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    async fn refresh_tools_if_needed(&self) -> Result<Option<Vec<Tool>>, AiError> {
        if !self.tool_state.tools_dirty.swap(false, Ordering::SeqCst) {
            return Ok(None);
        }
        match self.list_all_tools().await {
            Ok(tools) => Ok(Some(tools)),
            Err(error) => {
                self.tool_state.tools_dirty.store(true, Ordering::SeqCst);
                Err(error)
            }
        }
    }

    async fn start_or_restart_listener(&self, server_label: &str) {
        self.stop_listener().await;

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(run_streamable_http_listener(
            server_label.to_string(),
            self.url.clone(),
            self.client.clone(),
            self.oauth.clone(),
            self.state.clone(),
            self.tool_state.clone(),
            self.reinitialize_lock.clone(),
            shutdown_rx,
        ));
        *self.listener.lock().await = Some(RemoteHttpListener { shutdown_tx, task });
    }

    async fn stop_listener(&self) {
        if let Some(listener) = self.listener.lock().await.take() {
            let _ = listener.shutdown_tx.send(());
            let _ = listener.task.await;
        }
    }

    async fn call_tool(
        &self,
        remote_tool_name: &str,
        arguments: Map<String, Value>,
    ) -> Result<rmcp::model::CallToolResult, AiError> {
        let value = self
            .send_request_value(
                "tools/call",
                Some(json!({
                    "name": remote_tool_name,
                    "arguments": arguments,
                })),
            )
            .await?;
        serde_json::from_value(value)
            .map_err(|error| AiError::parse(format!("Invalid tools/call response: {}", error)))
    }

    async fn send_request_value(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, AiError> {
        self.send_request_internal(method, params, false, false)
            .await?
            .map(|(value, _)| value)
            .ok_or_else(|| AiError::api(format!("MCP request '{}' returned no response", method)))
    }

    async fn send_request_internal(
        &self,
        method: &str,
        params: Option<Value>,
        is_notification: bool,
        is_initialize: bool,
    ) -> Result<Option<(Value, Option<String>)>, AiError> {
        let mut attempted_oauth = false;
        let mut attempted_session_recovery = false;

        loop {
            let (request_id, session_id, protocol_version) = {
                let mut state = self.state.lock().await;
                let request_id = if is_notification {
                    None
                } else {
                    state.next_request_id = state.next_request_id.saturating_add(1);
                    Some(state.next_request_id.to_string())
                };
                (
                    request_id,
                    state.session_id.clone(),
                    state.protocol_version.clone(),
                )
            };

            let mut payload = json!({
                "jsonrpc": "2.0",
                "method": method,
            });
            if let Some(id) = &request_id {
                payload["id"] = Value::String(id.clone());
            }
            if let Some(params) = params.clone() {
                payload["params"] = params;
            }

            let mut builder = self
                .client
                .post(&self.url)
                .header("Accept", "application/json, text/event-stream")
                .header("Content-Type", "application/json");
            if !is_initialize {
                builder = builder.header("MCP-Protocol-Version", protocol_version);
            }
            if let Some(session_id) = &session_id {
                builder = builder.header("MCP-Session-Id", session_id);
            }
            if let Some(oauth) = &self.oauth {
                builder = oauth.attach_existing_token(builder).await?;
            }
            let response = builder
                .body(payload.to_string())
                .send()
                .await
                .map_err(|error| AiError::http(error.to_string()))?;

            if response.status() == reqwest::StatusCode::UNAUTHORIZED
                && !attempted_oauth
                && self.oauth.is_some()
            {
                attempted_oauth = true;
                self.oauth
                    .as_ref()
                    .unwrap()
                    .authorize_interactively()
                    .await?;
                continue;
            }

            if is_notification {
                if response.status().is_success()
                    || response.status() == reqwest::StatusCode::ACCEPTED
                    || response.status() == reqwest::StatusCode::NO_CONTENT
                {
                    return Ok(None);
                }
                return Err(AiError::api(format!(
                    "MCP notification '{}' failed with HTTP {}",
                    method,
                    response.status()
                )));
            }

            let session_header = response
                .headers()
                .get("MCP-Session-Id")
                .and_then(|value| value.to_str().ok())
                .map(|value| value.to_string());
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string();

            if !is_initialize
                && session_id.is_some()
                && response.status() == reqwest::StatusCode::NOT_FOUND
                && !attempted_session_recovery
            {
                attempted_session_recovery = true;
                Box::pin(self.reinitialize_after_session_expiry()).await?;
                continue;
            }

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(AiError::api(format!(
                    "MCP request '{}' failed with HTTP {}: {}",
                    method, status, body
                )));
            }

            let request_id = request_id.ok_or_else(|| {
                AiError::api(format!("MCP request '{}' missing request id", method))
            })?;
            if content_type.contains("text/event-stream") {
                let value = self
                    .collect_json_rpc_response_from_sse(response, &request_id)
                    .await?;
                return Ok(Some((extract_json_rpc_result(value)?, session_header)));
            }

            let value: Value = response
                .json()
                .await
                .map_err(|error| AiError::parse(format!("Invalid MCP JSON response: {}", error)))?;
            return Ok(Some((extract_json_rpc_result(value)?, session_header)));
        }
    }

    async fn collect_json_rpc_response_from_sse(
        &self,
        response: reqwest::Response,
        expected_id: &str,
    ) -> Result<Value, AiError> {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(next) = stream.next().await {
            let chunk = next.map_err(|error| AiError::http(error.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some((frame, remainder)) = split_sse_frame(&buffer) {
                buffer = remainder;
                let Some(event) = parse_sse_event(&frame) else {
                    continue;
                };
                if let Some(event_id) = &event.id {
                    *self.tool_state.last_event_id.lock().await = Some(event_id.clone());
                }
                if event.data.trim().is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(event.data.trim())
                    .map_err(|error| AiError::parse(format!("Invalid MCP SSE JSON: {}", error)))?;
                if value.get("id").map(json_rpc_id_string).as_deref() == Some(expected_id) {
                    return Ok(value);
                }
                if let Some(response_payload) =
                    handle_server_initiated_json_rpc(&value, &self.tool_state)?
                {
                    let _ = self.send_auxiliary_json_rpc_message(response_payload).await;
                }
            }
        }
        Err(AiError::api(
            "MCP SSE stream ended before the JSON-RPC response arrived".to_string(),
        ))
    }

    async fn send_auxiliary_json_rpc_message(&self, payload: Value) -> Result<(), AiError> {
        let (session_id, protocol_version) = {
            let state = self.state.lock().await;
            (state.session_id.clone(), state.protocol_version.clone())
        };
        let mut attempted_oauth = false;
        loop {
            let mut builder = self
                .client
                .post(&self.url)
                .header("Accept", "application/json, text/event-stream")
                .header("Content-Type", "application/json")
                .header("MCP-Protocol-Version", protocol_version.clone());
            if let Some(session_id) = &session_id {
                builder = builder.header("MCP-Session-Id", session_id);
            }
            if let Some(oauth) = &self.oauth {
                builder = oauth.attach_existing_token(builder).await?;
            }
            let response = builder
                .body(payload.to_string())
                .send()
                .await
                .map_err(|error| AiError::http(error.to_string()))?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED
                && !attempted_oauth
                && self.oauth.is_some()
            {
                attempted_oauth = true;
                self.oauth
                    .as_ref()
                    .unwrap()
                    .authorize_interactively()
                    .await?;
                continue;
            }
            if response.status().is_success()
                || response.status() == reqwest::StatusCode::ACCEPTED
                || response.status() == reqwest::StatusCode::NO_CONTENT
            {
                return Ok(());
            }
            return Err(AiError::api(format!(
                "MCP auxiliary message failed with HTTP {}",
                response.status()
            )));
        }
    }

    async fn reinitialize_after_session_expiry(&self) -> Result<(), AiError> {
        let _guard = self.reinitialize_lock.lock().await;
        {
            let mut state = self.state.lock().await;
            state.session_id = None;
        }
        self.initialize(&self.server_label).await
    }
}

impl LegacySseClient {
    async fn connect(
        server_label: &str,
        config: &McpServerConfig,
        client: reqwest::Client,
        oauth: Option<Arc<RemoteOAuthContext>>,
    ) -> Result<Self, AiError> {
        let base_url = config
            .url
            .clone()
            .ok_or_else(|| AiError::parse("MCP SSE server is missing 'url'".to_string()))?;
        let this = Self {
            base_url,
            client,
            oauth,
            endpoint_url: Mutex::new(String::new()),
            protocol_version: Mutex::new(MCP_PROTOCOL_VERSION_LEGACY_SSE.to_string()),
            next_request_id: AtomicU64::new(0),
            tool_state: Arc::new(RemoteToolCatalogState::default()),
            pending: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx: Mutex::new(None),
            reader_task: Mutex::new(None),
        };
        this.start_stream_and_initialize(server_label).await?;
        Ok(this)
    }

    async fn list_all_tools(&self) -> Result<Vec<Tool>, AiError> {
        let value = self
            .send_request_value("tools/list", Some(json!({})))
            .await?;
        let result: rmcp::model::ListToolsResult = serde_json::from_value(value)
            .map_err(|error| AiError::parse(format!("Invalid tools/list response: {}", error)))?;
        Ok(result.tools)
    }

    async fn call_tool(
        &self,
        remote_tool_name: &str,
        arguments: Map<String, Value>,
    ) -> Result<rmcp::model::CallToolResult, AiError> {
        let value = self
            .send_request_value(
                "tools/call",
                Some(json!({
                    "name": remote_tool_name,
                    "arguments": arguments,
                })),
            )
            .await?;
        serde_json::from_value(value)
            .map_err(|error| AiError::parse(format!("Invalid tools/call response: {}", error)))
    }

    async fn refresh_tools_if_needed(&self) -> Result<Option<Vec<Tool>>, AiError> {
        if !self.tool_state.tools_dirty.swap(false, Ordering::SeqCst) {
            return Ok(None);
        }
        match self.list_all_tools().await {
            Ok(tools) => Ok(Some(tools)),
            Err(error) => {
                self.tool_state.tools_dirty.store(true, Ordering::SeqCst);
                Err(error)
            }
        }
    }

    async fn close(&mut self) -> Result<(), AiError> {
        if let Some(shutdown) = self.shutdown_tx.lock().await.take() {
            let _ = shutdown.send(());
        }
        clear_pending_requests(&self.pending).await;
        if let Some(task) = self.reader_task.lock().await.take() {
            let _ = task.await;
        }
        clear_pending_requests(&self.pending).await;
        Ok(())
    }

    async fn start_stream_and_initialize(&self, server_label: &str) -> Result<(), AiError> {
        let (endpoint_tx, endpoint_rx) = oneshot::channel::<String>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        *self.shutdown_tx.lock().await = Some(shutdown_tx);

        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let server_label = server_label.to_string();
        let tool_state = self.tool_state.clone();
        let pending = self.pending.clone();
        let oauth = self.oauth.clone();
        let reader_task = tokio::spawn(async move {
            let _ = run_legacy_sse_reader(
                &server_label,
                &base_url,
                client,
                oauth,
                pending,
                tool_state,
                endpoint_tx,
                shutdown_rx,
            )
            .await;
        });
        *self.reader_task.lock().await = Some(reader_task);

        let endpoint = endpoint_rx.await.map_err(|_| {
            AiError::api("Legacy SSE stream closed before emitting the endpoint event".to_string())
        })?;
        *self.endpoint_url.lock().await = endpoint;

        let init = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION_LEGACY_SSE,
            "capabilities": {},
            "clientInfo": {
                "name": "connect_llm",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });
        let value = self.send_request_value("initialize", Some(init)).await?;
        let result: InitializeResultValue = serde_json::from_value(value).map_err(|error| {
            AiError::parse(format!("Invalid legacy SSE initialize response: {}", error))
        })?;
        validate_negotiated_protocol_version(
            &result.protocol_version,
            RemoteProtocolFlavor::LegacySse,
        )?;
        *self.protocol_version.lock().await = result.protocol_version;
        self.send_notification("notifications/initialized", None)
            .await?;
        Ok(())
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> Result<(), AiError> {
        self.post_message(method, params, None).await
    }

    async fn send_request_value(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, AiError> {
        let request_id = self
            .next_request_id
            .fetch_add(1, Ordering::SeqCst)
            .saturating_add(1)
            .to_string();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.pending.lock().await.insert(request_id.clone(), sender);
        if let Err(error) = self
            .post_message(method, params, Some(request_id.clone()))
            .await
        {
            remove_pending_request(&self.pending, &request_id).await;
            return Err(error);
        }
        let response = await_pending_legacy_sse_response(
            &self.pending,
            &request_id,
            method,
            receiver,
            Duration::from_secs(30),
        )
        .await?;
        extract_json_rpc_result(response)
    }

    async fn post_message(
        &self,
        method: &str,
        params: Option<Value>,
        request_id: Option<String>,
    ) -> Result<(), AiError> {
        let mut payload = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(id) = request_id {
            payload["id"] = Value::String(id);
        }
        if let Some(params) = params {
            payload["params"] = params;
        }

        let mut attempted_oauth = false;
        loop {
            let endpoint = self.endpoint_url.lock().await.clone();
            let mut builder = self
                .client
                .post(endpoint)
                .header("Content-Type", "application/json")
                .body(payload.to_string());
            if let Some(oauth) = &self.oauth {
                builder = oauth.attach_existing_token(builder).await?;
            }
            let response = builder
                .send()
                .await
                .map_err(|error| AiError::http(error.to_string()))?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED
                && !attempted_oauth
                && self.oauth.is_some()
            {
                attempted_oauth = true;
                self.oauth
                    .as_ref()
                    .unwrap()
                    .authorize_interactively()
                    .await?;
                continue;
            }
            if response.status().is_success()
                || response.status() == reqwest::StatusCode::ACCEPTED
                || response.status() == reqwest::StatusCode::NO_CONTENT
            {
                return Ok(());
            }
            return Err(AiError::api(format!(
                "Legacy SSE MCP request '{}' failed with HTTP {}",
                method,
                response.status()
            )));
        }
    }
}

#[derive(Debug)]
struct ParsedSseEvent {
    event: Option<String>,
    id: Option<String>,
    retry_millis: Option<u64>,
    data: String,
}

async fn run_legacy_sse_reader(
    _server_label: &str,
    base_url: &str,
    client: reqwest::Client,
    oauth: Option<Arc<RemoteOAuthContext>>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    tool_state: Arc<RemoteToolCatalogState>,
    endpoint_tx: oneshot::Sender<String>,
    mut shutdown_rx: oneshot::Receiver<()>,
) -> Result<(), AiError> {
    let result = async {
        let mut attempted_oauth = false;
        let response = loop {
            let mut builder = client.get(base_url).header("Accept", "text/event-stream");
            if let Some(oauth) = &oauth {
                builder = oauth.attach_existing_token(builder).await?;
            }
            let response = builder
                .send()
                .await
                .map_err(|error| AiError::http(error.to_string()))?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED
                && !attempted_oauth
                && oauth.is_some()
            {
                attempted_oauth = true;
                oauth.as_ref().unwrap().authorize_interactively().await?;
                continue;
            }
            if !response.status().is_success() {
                return Err(AiError::api(format!(
                    "Legacy SSE stream failed with HTTP {}",
                    response.status()
                )));
            }
            break response;
        };

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut endpoint_tx = Some(endpoint_tx);
        let mut endpoint_url: Option<String> = None;

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                next = stream.next() => {
                    let Some(chunk) = next else { break; };
                    let chunk = chunk.map_err(|error| AiError::http(error.to_string()))?;
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                    while let Some((frame, remainder)) = split_sse_frame(&buffer) {
                        buffer = remainder;
                        let Some(event) = parse_sse_event(&frame) else {
                            continue;
                        };
                        if let Some(event_id) = &event.id {
                            *tool_state.last_event_id.lock().await = Some(event_id.clone());
                        }
                        if event.event.as_deref() == Some("endpoint") {
                            let endpoint = resolve_legacy_endpoint(base_url, event.data.trim());
                            endpoint_url = Some(endpoint.clone());
                            if let Some(sender) = endpoint_tx.take() {
                                let _ = sender.send(endpoint);
                            }
                            continue;
                        }
                        if event.data.trim().is_empty() {
                            continue;
                        }
                        let value: Value = match serde_json::from_str(event.data.trim()) {
                            Ok(value) => value,
                            Err(_) => continue,
                        };
                        if let Some(id) = value.get("id").map(json_rpc_id_string) {
                            if let Some(sender) = pending.lock().await.remove(&id) {
                                let _ = sender.send(value);
                                continue;
                            }
                        }
                        if let Some(response_payload) =
                            handle_server_initiated_json_rpc(&value, &tool_state)?
                        {
                            if let Some(endpoint) = &endpoint_url {
                                let _ = send_legacy_sse_auxiliary_message(
                                    &client,
                                    oauth.as_ref(),
                                    endpoint,
                                    response_payload,
                                )
                                .await;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
    .await;
    clear_pending_requests(&pending).await;
    result
}

async fn run_streamable_http_listener(
    _server_label: String,
    url: String,
    client: reqwest::Client,
    oauth: Option<Arc<RemoteOAuthContext>>,
    state: Arc<Mutex<RemoteHttpState>>,
    tool_state: Arc<RemoteToolCatalogState>,
    _reinitialize_lock: Arc<Mutex<()>>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    let mut reconnect_delay = Duration::from_secs(1);

    loop {
        let (session_id, protocol_version) = {
            let state = state.lock().await;
            (state.session_id.clone(), state.protocol_version.clone())
        };
        let last_event_id = tool_state.last_event_id.lock().await.clone();

        let Some(session_id) = session_id else {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                _ = sleep(Duration::from_millis(250)) => continue,
            }
        };

        let mut attempted_oauth = false;
        let response = loop {
            let mut builder = client
                .get(&url)
                .header("Accept", "text/event-stream")
                .header("MCP-Protocol-Version", protocol_version.clone())
                .header("MCP-Session-Id", session_id.clone());
            if let Some(last_event_id) = &last_event_id {
                builder = builder.header("Last-Event-ID", last_event_id);
            }
            if let Some(oauth) = &oauth {
                match oauth.attach_existing_token(builder).await {
                    Ok(with_auth) => builder = with_auth,
                    Err(_) if attempted_oauth => break None,
                    Err(_) => {
                        attempted_oauth = true;
                        if oauth.authorize_interactively().await.is_err() {
                            break None;
                        }
                        continue;
                    }
                }
            }
            match builder.send().await {
                Ok(response) => {
                    if response.status() == reqwest::StatusCode::UNAUTHORIZED
                        && !attempted_oauth
                        && oauth.is_some()
                    {
                        attempted_oauth = true;
                        if oauth
                            .as_ref()
                            .unwrap()
                            .authorize_interactively()
                            .await
                            .is_err()
                        {
                            break None;
                        }
                        continue;
                    }
                    break Some(response);
                }
                Err(_) => break None,
            }
        };

        let Some(response) = response else {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                _ = sleep(reconnect_delay) => continue,
            }
        };

        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED
            || response.status() == reqwest::StatusCode::NOT_IMPLEMENTED
        {
            break;
        }
        if !response.status().is_success() {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                _ = sleep(reconnect_delay) => continue,
            }
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut stream_retry = reconnect_delay;

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => return,
                next = stream.next() => {
                    let Some(chunk) = next else { break; };
                    let chunk = match chunk {
                        Ok(chunk) => chunk,
                        Err(_) => break,
                    };
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                    while let Some((frame, remainder)) = split_sse_frame(&buffer) {
                        buffer = remainder;
                        let Some(event) = parse_sse_event(&frame) else {
                            continue;
                        };
                        if let Some(event_id) = &event.id {
                            *tool_state.last_event_id.lock().await = Some(event_id.clone());
                        }
                        if let Some(retry_millis) = event.retry_millis {
                            stream_retry = Duration::from_millis(retry_millis.max(1));
                        }
                        if event.data.trim().is_empty() {
                            continue;
                        }
                        let value: Value = match serde_json::from_str(event.data.trim()) {
                            Ok(value) => value,
                            Err(_) => continue,
                        };
                        if let Some(response_payload) =
                            handle_server_initiated_json_rpc(&value, &tool_state).unwrap_or(None)
                        {
                            let _ = send_streamable_http_auxiliary_message(
                                &client,
                                oauth.as_ref(),
                                &url,
                                &protocol_version,
                                Some(&session_id),
                                response_payload,
                            )
                            .await;
                        }
                    }
                }
            }
        }

        reconnect_delay = stream_retry;
        tokio::select! {
            _ = &mut shutdown_rx => break,
            _ = sleep(reconnect_delay) => {}
        }
    }
}

fn parse_sse_event(frame: &str) -> Option<ParsedSseEvent> {
    let mut event = None;
    let mut id = None;
    let mut retry_millis = None;
    let mut data = Vec::new();
    for raw_line in frame.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("id:") {
            id = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("retry:") {
            retry_millis = value.trim().parse::<u64>().ok();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start().to_string());
        }
    }
    if event.is_none() && id.is_none() && retry_millis.is_none() && data.is_empty() {
        return None;
    }
    Some(ParsedSseEvent {
        event,
        id,
        retry_millis,
        data: data.join("\n"),
    })
}

fn split_sse_frame(buffer: &str) -> Option<(String, String)> {
    if let Some(index) = buffer.find("\r\n\r\n") {
        let frame = buffer[..index].to_string();
        let remainder = buffer[index + 4..].to_string();
        return Some((frame, remainder));
    }
    if let Some(index) = buffer.find("\n\n") {
        let frame = buffer[..index].to_string();
        let remainder = buffer[index + 2..].to_string();
        return Some((frame, remainder));
    }
    None
}

fn resolve_legacy_endpoint(base_url: &str, endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else if let Ok(base) = reqwest::Url::parse(base_url) {
        base.join(endpoint)
            .map(|value| value.to_string())
            .unwrap_or_else(|_| endpoint.to_string())
    } else {
        endpoint.to_string()
    }
}

fn handle_server_initiated_json_rpc(
    value: &Value,
    tool_state: &RemoteToolCatalogState,
) -> Result<Option<Value>, AiError> {
    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return Ok(None);
    };

    if value.get("id").is_none() {
        if method == "notifications/tools/list_changed" {
            tool_state.tools_dirty.store(true, Ordering::SeqCst);
        }
        return Ok(None);
    }

    let request_id = value
        .get("id")
        .cloned()
        .ok_or_else(|| AiError::parse("Server request is missing JSON-RPC id".to_string()))?;

    let response = match method {
        "ping" => json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }),
        "roots/list" => json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Roots not supported",
                "data": {
                    "reason": "connect_llm does not advertise the roots capability"
                }
            }
        }),
        unsupported => json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Method not found",
                "data": {
                    "reason": format!("connect_llm does not support server-initiated method '{}'", unsupported)
                }
            }
        }),
    };

    Ok(Some(response))
}

async fn send_legacy_sse_auxiliary_message(
    client: &reqwest::Client,
    oauth: Option<&Arc<RemoteOAuthContext>>,
    endpoint: &str,
    payload: Value,
) -> Result<(), AiError> {
    let mut attempted_oauth = false;
    loop {
        let mut builder = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .body(payload.to_string());
        if let Some(oauth) = oauth {
            builder = oauth.attach_existing_token(builder).await?;
        }
        let response = builder
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            && !attempted_oauth
            && oauth.is_some()
        {
            attempted_oauth = true;
            oauth.unwrap().authorize_interactively().await?;
            continue;
        }
        if response.status().is_success()
            || response.status() == reqwest::StatusCode::ACCEPTED
            || response.status() == reqwest::StatusCode::NO_CONTENT
        {
            return Ok(());
        }
        return Err(AiError::api(format!(
            "Legacy SSE auxiliary message failed with HTTP {}",
            response.status()
        )));
    }
}

async fn send_streamable_http_auxiliary_message(
    client: &reqwest::Client,
    oauth: Option<&Arc<RemoteOAuthContext>>,
    url: &str,
    protocol_version: &str,
    session_id: Option<&str>,
    payload: Value,
) -> Result<(), AiError> {
    let mut attempted_oauth = false;
    loop {
        let mut builder = client
            .post(url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header("MCP-Protocol-Version", protocol_version);
        if let Some(session_id) = session_id {
            builder = builder.header("MCP-Session-Id", session_id);
        }
        if let Some(oauth) = oauth {
            builder = oauth.attach_existing_token(builder).await?;
        }
        let response = builder
            .body(payload.to_string())
            .send()
            .await
            .map_err(|error| AiError::http(error.to_string()))?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            && !attempted_oauth
            && oauth.is_some()
        {
            attempted_oauth = true;
            oauth.unwrap().authorize_interactively().await?;
            continue;
        }
        if response.status().is_success()
            || response.status() == reqwest::StatusCode::ACCEPTED
            || response.status() == reqwest::StatusCode::NO_CONTENT
        {
            return Ok(());
        }
        return Err(AiError::api(format!(
            "Streamable HTTP auxiliary message failed with HTTP {}",
            response.status()
        )));
    }
}

fn extract_json_rpc_result(value: Value) -> Result<Value, AiError> {
    let response: JsonRpcResponseEnvelope = serde_json::from_value(value)
        .map_err(|error| AiError::parse(format!("Invalid JSON-RPC response: {}", error)))?;
    if let Some(error) = response.error {
        let detail = error
            .data
            .map(|data| format!(": {}", data))
            .unwrap_or_default();
        return Err(AiError::api(format!(
            "MCP JSON-RPC error {} {}{}",
            error.code, error.message, detail
        )));
    }
    response
        .result
        .ok_or_else(|| AiError::api("MCP JSON-RPC response did not include a result".to_string()))
}

fn json_rpc_id_string(id: &Value) -> String {
    match id {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        other => other.to_string(),
    }
}

fn build_remote_reqwest_client(
    headers: &BTreeMap<String, String>,
) -> Result<reqwest::Client, AiError> {
    let mut default_headers = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| AiError::parse(format!("Invalid MCP header '{}': {}", name, error)))?;
        let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
            AiError::parse(format!(
                "Invalid MCP header value for '{}': {}",
                name, error
            ))
        })?;
        default_headers.insert(header_name, header_value);
    }
    reqwest::Client::builder()
        .default_headers(default_headers)
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| AiError::http(error.to_string()))
}

fn oauth_store_path(server_label: &str, config: &McpServerConfig) -> PathBuf {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(server_label.as_bytes());
    if let Some(url) = &config.url {
        hasher.update(url.as_bytes());
    }
    if let Some(server_type) = &config.server_type {
        hasher.update(server_type.as_bytes());
    }
    let digest = format!("{:x}", hasher.finalize());
    oauth_store_root().join(format!("{}.json", digest))
}

fn oauth_store_root() -> PathBuf {
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".connect_llm")
        .join("mcp_oauth")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
}

async fn fetch_authorization_metadata(url: &str) -> Result<AuthorizationMetadata, AiError> {
    if !url.starts_with("https://") {
        return Err(AiError::parse(format!(
            "authServerMetadataUrl must use https:// (got: {})",
            url
        )));
    }
    reqwest::Client::new()
        .get(url)
        .send()
        .await
        .map_err(|error| AiError::http(error.to_string()))?
        .json::<AuthorizationMetadata>()
        .await
        .map_err(|error| AiError::parse(format!("Invalid OAuth metadata: {}", error)))
}

fn open_url_in_browser_local(url: &str) -> Result<(), AiError> {
    #[cfg(any(target_os = "windows", target_os = "macos", unix))]
    {
        browser_open_command_local(url).spawn().map_err(|error| {
            AiError::api(format!(
                "Failed to open browser automatically: {}. Open this URL manually: {}",
                error, url
            ))
        })?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err(AiError::api(format!(
        "Automatic browser opening is not supported on this platform. Open this URL manually: {}",
        url
    )))
}

#[cfg(target_os = "windows")]
fn browser_open_command_local(url: &str) -> std::process::Command {
    let mut command = std::process::Command::new("explorer");
    command.arg(url);
    command
}

#[cfg(target_os = "macos")]
fn browser_open_command_local(url: &str) -> std::process::Command {
    let mut command = std::process::Command::new("open");
    command.arg(url);
    command
}

#[cfg(all(unix, not(target_os = "macos")))]
fn browser_open_command_local(url: &str) -> std::process::Command {
    let mut command = std::process::Command::new("xdg-open");
    command.arg(url);
    command
}

fn wait_for_oauth_callback(
    listener: TcpListener,
    timeout_duration: Duration,
) -> Result<OAuthCallbackResult, AiError> {
    listener
        .set_nonblocking(true)
        .map_err(|error| AiError::http(error.to_string()))?;
    let (sender, receiver) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let handle = thread::spawn(move || {
        let result = loop {
            if worker_stop.load(Ordering::SeqCst) {
                return;
            }
            match listener.accept() {
                Ok((stream, _)) => break handle_oauth_callback_stream(stream),
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(25));
                }
                Err(error) => break Err(AiError::http(error.to_string())),
            }
        };
        let _ = sender.send(result);
    });

    let result = receiver
        .recv_timeout(timeout_duration)
        .map_err(|_| AiError::api("Timed out waiting for the browser OAuth callback.".to_string()));
    stop.store(true, Ordering::SeqCst);
    let _ = handle.join();
    result?
}

fn handle_oauth_callback_stream(
    mut stream: std::net::TcpStream,
) -> Result<OAuthCallbackResult, AiError> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|error| AiError::http(error.to_string()))?;
    let mut buffer = [0u8; 8192];
    let read_len = stream
        .read(&mut buffer)
        .map_err(|error| AiError::http(error.to_string()))?;
    let request = String::from_utf8_lossy(&buffer[..read_len]);
    let request_line = request.lines().next().unwrap_or_default();
    let target = request_line.split_whitespace().nth(1).unwrap_or("/");
    let url = reqwest::Url::parse(&format!("http://localhost{}", target))
        .map_err(|error| AiError::parse(error.to_string()))?;
    let code = url
        .query_pairs()
        .find(|(key, _)| key == "code")
        .map(|(_, value)| value.to_string())
        .ok_or_else(|| AiError::api("OAuth callback did not contain a code".to_string()))?;
    let state = url
        .query_pairs()
        .find(|(key, _)| key == "state")
        .map(|(_, value)| value.to_string())
        .ok_or_else(|| AiError::api("OAuth callback did not contain a state".to_string()))?;
    let body = "<html><body><h1>Authorization Successful</h1><p>You can close this window.</p></body></html>";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    Ok(OAuthCallbackResult { code, state })
}

fn map_auth_error(error: AuthError) -> AiError {
    AiError::api(format!("MCP OAuth error: {}", error))
}

fn validate_negotiated_protocol_version(
    protocol_version: &str,
    flavor: RemoteProtocolFlavor,
) -> Result<(), AiError> {
    let supported = match flavor {
        RemoteProtocolFlavor::StreamableHttp => SUPPORTED_STREAMABLE_HTTP_PROTOCOL_VERSIONS,
        RemoteProtocolFlavor::LegacySse => SUPPORTED_LEGACY_SSE_PROTOCOL_VERSIONS,
    };
    if supported.contains(&protocol_version) {
        Ok(())
    } else {
        Err(AiError::api(format!(
            "Unsupported MCP protocol version negotiated for {}: {} (supported: {})",
            match flavor {
                RemoteProtocolFlavor::StreamableHttp => "streamable-http",
                RemoteProtocolFlavor::LegacySse => "legacy-sse",
            },
            protocol_version,
            supported.join(", ")
        )))
    }
}

fn detect_transport(config: &McpServerConfig) -> Result<McpTransport, AiError> {
    if let Some(server_type) = config.server_type.as_deref() {
        return match server_type {
            "stdio" => Ok(McpTransport::Stdio),
            "http" | "streamable_http" => Ok(McpTransport::StreamableHttp),
            "sse" => Ok(McpTransport::LegacySse),
            "ws" => Err(AiError::parse(
                "Claude-compatible MCP type 'ws' is not supported by this build yet".to_string(),
            )),
            "sdk" => Err(AiError::parse(
                "Claude-compatible MCP type 'sdk' is not supported by this build yet".to_string(),
            )),
            "claudeai-proxy" => Err(AiError::parse(
                "Claude-compatible MCP type 'claudeai-proxy' is not supported by this build yet"
                    .to_string(),
            )),
            other => Err(AiError::parse(format!(
                "Unsupported Claude-compatible MCP type '{}'",
                other
            ))),
        };
    }
    if let Some(transport) = config.transport.as_deref() {
        return match transport {
            "stdio" => Ok(McpTransport::Stdio),
            "http" | "streamable_http" => Ok(McpTransport::StreamableHttp),
            "sse" => Ok(McpTransport::LegacySse),
            other => Err(AiError::parse(format!(
                "Unsupported MCP transport '{}'",
                other
            ))),
        };
    }

    match (config.command.is_some(), config.url.is_some()) {
        (true, false) => Ok(McpTransport::Stdio),
        (false, true) => Ok(McpTransport::StreamableHttp),
        (true, true) => Err(AiError::parse(
            "MCP server config cannot specify both 'command' and 'url' without an explicit transport"
                .to_string(),
        )),
        (false, false) => Err(AiError::parse(
            "MCP server config must specify either 'command' or 'url'".to_string(),
        )),
    }
}

fn validate_supported_config(config: &McpServerConfig) -> Result<(), AiError> {
    if let Some(oauth) = &config.oauth {
        if oauth.xaa.unwrap_or(false) {
            return Err(AiError::parse(
                "Claude-compatible MCP OAuth xaa is not supported by this build yet".to_string(),
            ));
        }
    }
    Ok(())
}

async fn build_remote_headers(
    config: &McpServerConfig,
    server_label: Option<&str>,
) -> Result<BTreeMap<String, String>, AiError> {
    let mut headers = config.headers.clone();
    apply_configured_auth_header(&mut headers, config.auth_header.as_deref())?;
    if let Some(dynamic_headers) = run_headers_helper(config, server_label).await? {
        headers.extend(dynamic_headers);
    }
    Ok(headers)
}

fn apply_configured_auth_header(
    headers: &mut BTreeMap<String, String>,
    auth_header: Option<&str>,
) -> Result<(), AiError> {
    let Some(auth_header) = auth_header else {
        return Ok(());
    };
    if let Some((existing_name, existing_value)) = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("Authorization"))
    {
        if existing_value != auth_header {
            return Err(AiError::parse(format!(
                "MCP remote config cannot set both '{}' and authHeader/authorization",
                existing_name
            )));
        }
        return Ok(());
    }
    headers.insert("Authorization".to_string(), auth_header.to_string());
    Ok(())
}

fn oauth_metadata_source(config: &McpOAuthConfig) -> OAuthMetadataSource {
    if config.auth_server_metadata_url.is_some() {
        OAuthMetadataSource::Configured
    } else {
        OAuthMetadataSource::Discover
    }
}

async fn run_headers_helper(
    config: &McpServerConfig,
    server_label: Option<&str>,
) -> Result<Option<BTreeMap<String, String>>, AiError> {
    let Some(helper) = config.headers_helper.as_deref() else {
        return Ok(None);
    };

    let mut command = shell_command_for_platform(helper);
    if let Some(label) = server_label {
        command.env("CLAUDE_CODE_MCP_SERVER_NAME", label);
    }
    if let Some(url) = config.url.as_deref() {
        command.env("CLAUDE_CODE_MCP_SERVER_URL", url);
    }
    let output = timeout(Duration::from_secs(10), command.output())
        .await
        .map_err(|_| AiError::api("MCP headersHelper timed out after 10 seconds".to_string()))?
        .map_err(|error| AiError::api(format!("Failed to execute MCP headersHelper: {}", error)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit code {:?}", output.status.code())
        } else {
            stderr
        };
        return Err(AiError::api(format!(
            "MCP headersHelper failed: {}",
            detail
        )));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        AiError::parse(format!(
            "MCP headersHelper returned non-UTF-8 output: {}",
            error
        ))
    })?;
    let parsed: Value = serde_json::from_str(stdout.trim()).map_err(|error| {
        AiError::parse(format!(
            "MCP headersHelper must return a JSON object: {}",
            error
        ))
    })?;
    let object = parsed
        .as_object()
        .ok_or_else(|| AiError::parse("MCP headersHelper must return a JSON object".to_string()))?;

    let mut headers = BTreeMap::new();
    for (key, value) in object {
        let value = value.as_str().ok_or_else(|| {
            AiError::parse(format!(
                "MCP headersHelper value for '{}' must be a string",
                key
            ))
        })?;
        headers.insert(key.clone(), value.to_string());
    }
    Ok(Some(headers))
}

fn shell_command_for_platform(command_text: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("cmd");
        command.arg("/C").arg(command_text);
        command
    }
    #[cfg(not(windows))]
    {
        let mut command = Command::new("sh");
        command.arg("-lc").arg(command_text);
        command
    }
}

fn looks_like_filesystem_path(value: &str) -> bool {
    value.contains('\\') || value.contains('/') || value.starts_with('.')
}

fn expand_env_vars_in_string(
    value: &str,
    server_label: &str,
    field: &str,
) -> Result<String, AiError> {
    let mut missing = Vec::new();
    let mut expanded = String::with_capacity(value.len());
    let mut cursor = 0usize;

    while let Some(relative_start) = value[cursor..].find("${") {
        let start = cursor + relative_start;
        expanded.push_str(&value[cursor..start]);
        let rest = &value[start + 2..];
        let end = rest.find('}').ok_or_else(|| {
            AiError::parse(format!(
                "Invalid environment variable expression in MCP server '{}' field '{}'",
                server_label, field
            ))
        })?;
        let raw = &rest[..end];
        let (name, default) = match raw.split_once(":-") {
            Some((name, default)) => (name, Some(default)),
            None => (raw, None),
        };
        match std::env::var(name) {
            Ok(current) => expanded.push_str(&current),
            Err(_) => {
                if let Some(default) = default {
                    expanded.push_str(default);
                } else {
                    missing.push(name.to_string());
                    expanded.push_str("${");
                    expanded.push_str(raw);
                    expanded.push('}');
                }
            }
        }
        cursor = start + 2 + end + 1;
    }

    expanded.push_str(&value[cursor..]);

    if missing.is_empty() {
        Ok(expanded)
    } else {
        Err(AiError::parse(format!(
            "Missing environment variables in MCP server '{}' field '{}': {}",
            server_label,
            field,
            missing.join(", ")
        )))
    }
}

fn build_tool_description(
    server_label: &str,
    server_description: Option<&str>,
    remote_tool_name: &str,
    tool_description: Option<&str>,
) -> String {
    let mut description = format!(
        "MCP tool from server '{}' (remote name '{}').",
        server_label, remote_tool_name
    );
    if let Some(server_description) = server_description {
        if !server_description.trim().is_empty() {
            description.push(' ');
            description.push_str(server_description.trim());
        }
    }
    if let Some(tool_description) = tool_description {
        if !tool_description.trim().is_empty() {
            description.push(' ');
            description.push_str(tool_description.trim());
        }
    }
    description
}

fn allocate_tool_alias(
    server_label: &str,
    remote_tool_name: &str,
    used_aliases: &mut HashSet<String>,
) -> String {
    let server = sanitize_alias_segment(server_label, "server");
    let tool = sanitize_alias_segment(remote_tool_name, "tool");
    let base = format!("mcp_{}_{}", server, tool);
    if used_aliases.insert(base.clone()) {
        return base;
    }

    let mut index = 2usize;
    loop {
        let candidate = format!("{}_{}", base, index);
        if used_aliases.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn sanitize_alias_segment(input: &str, fallback: &str) -> String {
    let sanitized = input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .split('_')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if sanitized.is_empty() {
        fallback.to_string()
    } else {
        sanitized
    }
}

fn assistant_message_from_response(response: &ChatResponse) -> Message {
    let mut message =
        Message::assistant(response.content.clone()).with_tool_calls(response.tool_calls.clone());
    if let Some(thinking) = response.thinking.clone() {
        message = message.with_thinking(thinking);
    }
    message
}

#[derive(Default)]
struct StreamResponseBuilder {
    model: String,
    content: String,
    thinking_text: String,
    thinking_signature: Option<String>,
    tool_calls: Vec<PendingToolCall>,
    images: Vec<GeneratedImage>,
    seen_image_keys: HashSet<String>,
    debug_request: Option<String>,
    debug_responses: Vec<String>,
}

impl StreamResponseBuilder {
    fn new(model: String) -> Self {
        Self {
            model,
            ..Self::default()
        }
    }

    fn ingest(&mut self, chunk: &StreamChunk) {
        self.content.push_str(&chunk.delta);

        if let Some(thinking_delta) = &chunk.thinking_delta {
            self.thinking_text.push_str(thinking_delta);
        }
        if let Some(signature) = &chunk.thinking_signature {
            self.thinking_signature = Some(signature.clone());
        }

        for tool_call_delta in &chunk.tool_call_deltas {
            while self.tool_calls.len() <= tool_call_delta.index {
                self.tool_calls.push(PendingToolCall::default());
            }
            if let Some(tool_call) = self.tool_calls.get_mut(tool_call_delta.index) {
                tool_call.apply(tool_call_delta);
            }
        }

        for image in &chunk.images {
            let key = image.dedup_key();
            if self.seen_image_keys.insert(key) {
                self.images.push(image.clone());
            }
        }

        if let Some(debug) = &chunk.debug {
            if self.debug_request.is_none() {
                self.debug_request = debug.request.clone();
            }
            if let Some(response) = &debug.response {
                self.debug_responses.push(response.clone());
            }
        }
    }

    fn finish(self) -> ChatResponse {
        ChatResponse {
            id: "stream".to_string(),
            content: self.content,
            model: self.model,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            thinking: if !self.thinking_text.is_empty() {
                Some(ThinkingOutput {
                    text: Some(self.thinking_text),
                    signature: self.thinking_signature,
                    redacted: None,
                })
            } else if self.thinking_signature.is_some() {
                Some(ThinkingOutput {
                    text: None,
                    signature: self.thinking_signature,
                    redacted: None,
                })
            } else {
                None
            },
            images: self.images,
            tool_calls: self
                .tool_calls
                .into_iter()
                .enumerate()
                .filter_map(|(index, pending)| pending.finish(index))
                .collect(),
            debug: if self.debug_request.is_some() || !self.debug_responses.is_empty() {
                Some(DebugTrace {
                    request: self.debug_request,
                    response: if self.debug_responses.is_empty() {
                        None
                    } else {
                        Some(self.debug_responses.join("\n"))
                    },
                })
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl PendingToolCall {
    fn apply(&mut self, delta: &ToolCallDelta) {
        if let Some(id) = &delta.id {
            self.id = Some(id.clone());
        }
        if let Some(name) = &delta.name {
            self.name = Some(name.clone());
        }
        if let Some(arguments) = &delta.arguments {
            self.arguments.push_str(arguments);
        }
    }

    fn finish(self, index: usize) -> Option<ToolCall> {
        self.name.map(|name| ToolCall {
            id: self.id.unwrap_or_else(|| format!("tool-call-{}", index)),
            name,
            arguments: serde_json::from_str(&self.arguments)
                .unwrap_or_else(|_| Value::String(self.arguments)),
        })
    }
}

fn value_as_json_object(value: &Value) -> Result<Map<String, Value>, AiError> {
    match value {
        Value::Object(object) => Ok(object.clone()),
        Value::String(text) => try_parse_json_object_string(text).ok_or_else(|| {
            AiError::parse(format!(
                "MCP tool arguments must be a JSON object, got {}",
                value
            ))
        }),
        other => Err(AiError::parse(format!(
            "MCP tool arguments must be a JSON object, got {}",
            other
        ))),
    }
}

fn try_parse_json_object_string(text: &str) -> Option<Map<String, Value>> {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.starts_with('{') {
        return None;
    }

    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(trimmed) {
        return Some(object);
    }

    let repaired = repair_json_object_fragment(trimmed)?;
    match serde_json::from_str::<Value>(&repaired).ok()? {
        Value::Object(object) => Some(object),
        _ => None,
    }
}

fn repair_json_object_fragment(text: &str) -> Option<String> {
    let mut repaired = text.to_string();
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escape = false;

    for ch in text.chars() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                let expected = stack.pop()?;
                if ch != expected {
                    return None;
                }
            }
            _ => {}
        }
    }

    if in_string {
        repaired.push('"');
    }
    while let Some(ch) = stack.pop() {
        repaired.push(ch);
    }
    Some(repaired)
}

fn call_tool_result_to_value(result: &rmcp::model::CallToolResult) -> Value {
    if let Some(structured) = &result.structured_content {
        return structured.clone();
    }

    if result.content.len() == 1 {
        if let Content {
            raw: rmcp::model::RawContent::Text(text),
            ..
        } = &result.content[0]
        {
            return parse_json_like_text_value(&text.text)
                .unwrap_or_else(|| Value::String(text.text.clone()));
        }
    }

    json!({
        "content": result.content.iter().map(content_to_value).collect::<Vec<_>>(),
        "is_error": result.is_error.unwrap_or(false),
    })
}

async fn remove_pending_request(
    pending: &Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    request_id: &str,
) {
    pending.lock().await.remove(request_id);
}

async fn clear_pending_requests(pending: &Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>) {
    pending.lock().await.clear();
}

async fn await_pending_legacy_sse_response(
    pending: &Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    request_id: &str,
    method: &str,
    receiver: oneshot::Receiver<Value>,
    timeout_duration: Duration,
) -> Result<Value, AiError> {
    match timeout(timeout_duration, receiver).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(_)) => {
            remove_pending_request(pending, request_id).await;
            Err(AiError::api(format!(
                "Legacy SSE response channel closed for '{}'",
                method
            )))
        }
        Err(_) => {
            remove_pending_request(pending, request_id).await;
            Err(AiError::api(format!(
                "Timed out waiting for legacy SSE response to '{}'",
                method
            )))
        }
    }
}

fn parse_json_like_text_value(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    let first = trimmed.chars().next()?;
    if !matches!(first, '{' | '[' | '"' | '-' | '0'..='9' | 't' | 'f' | 'n') {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

fn content_to_value(content: &Content) -> Value {
    match &content.raw {
        rmcp::model::RawContent::Text(text) => json!({
            "type": "text",
            "text": text.text,
        }),
        rmcp::model::RawContent::Image(image) => json!({
            "type": "image",
            "data_base64": image.data,
            "mime_type": image.mime_type,
        }),
        rmcp::model::RawContent::Audio(audio) => json!({
            "type": "audio",
            "data_base64": audio.data,
            "mime_type": audio.mime_type,
        }),
        rmcp::model::RawContent::Resource(resource) => {
            serde_json::to_value(resource).unwrap_or_default()
        }
        rmcp::model::RawContent::ResourceLink(resource) => {
            serde_json::to_value(resource).unwrap_or_default()
        }
    }
}

fn mcp_service_error(error: impl std::fmt::Display) -> AiError {
    AiError::api(format!("MCP error: {}", error))
}

#[derive(Debug)]
enum McpTransport {
    Stdio,
    StreamableHttp,
    LegacySse,
}

#[cfg(test)]
mod tests {
    use super::{
        McpBridge, McpConfig, McpManagedChatResponse, McpToolLoopConfig, PendingMcpServer,
    };
    use crate::ai::{
        AiAuth, AiClient, AiConfig, AiError, AiProvider, ChatRequest, ChatResponse, Message,
        StreamChunk, ToolCall, ToolCallDelta, Usage,
    };
    use futures_util::{
        StreamExt,
        stream::{self, BoxStream},
    };
    use rmcp::{
        ServerHandler, ServiceExt,
        model::{CallToolRequestParams, CallToolResult, ListToolsResult, Tool},
    };
    use serde_json::{Map, Value, json};
    use std::{
        collections::HashMap,
        ffi::OsStr,
        fs,
        io::{Read, Write},
        path::PathBuf,
        sync::{Arc, Mutex},
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    #[derive(Clone)]
    struct MockAiClient {
        calls: Arc<Mutex<Vec<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl AiClient for MockAiClient {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
            self.calls.lock().unwrap().push(request.clone());
            match self.calls.lock().unwrap().len() {
                1 => Ok(ChatResponse {
                    id: "first".to_string(),
                    content: String::new(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    images: Vec::new(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "mcp_calc_sum".to_string(),
                        arguments: json!({"a": 2, "b": 3}),
                    }],
                    debug: None,
                }),
                _ => {
                    let tool_result = request
                        .messages
                        .last()
                        .and_then(|message| message.tool_result_value().cloned())
                        .unwrap_or_default();
                    Ok(ChatResponse {
                        id: "final".to_string(),
                        content: format!("tool returned {}", tool_result),
                        model: request.model,
                        usage: Usage {
                            input_tokens: 10,
                            output_tokens: 10,
                        },
                        thinking: None,
                        images: Vec::new(),
                        tool_calls: Vec::new(),
                        debug: None,
                    })
                }
            }
        }

        fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
            stream::empty().boxed()
        }

        fn config(&self) -> &AiConfig {
            static CONFIG: std::sync::OnceLock<AiConfig> = std::sync::OnceLock::new();
            CONFIG.get_or_init(|| {
                AiConfig::new(AiProvider::OpenAi)
                    .with_auth(AiAuth::BearerToken("test".to_string()))
                    .with_base_url("https://example.com")
                    .with_default_model("mock")
            })
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec!["mock".to_string()])
        }
    }

    #[derive(Clone)]
    struct LoopingAiClient {
        calls: Arc<Mutex<Vec<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl AiClient for LoopingAiClient {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
            self.calls.lock().unwrap().push(request.clone());
            if request.tools.is_empty() {
                return Ok(ChatResponse {
                    id: "final".to_string(),
                    content: "I could not get search results from the tool.".to_string(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                    debug: None,
                });
            }

            Ok(ChatResponse {
                id: "loop".to_string(),
                content: String::new(),
                model: request.model,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 10,
                },
                thinking: None,
                images: Vec::new(),
                tool_calls: vec![ToolCall {
                    id: format!("call_{}", self.calls.lock().unwrap().len()),
                    name: "mcp_calc_sum".to_string(),
                    arguments: json!({"a": 2, "b": 3}),
                }],
                debug: None,
            })
        }

        fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
            stream::empty().boxed()
        }

        fn config(&self) -> &AiConfig {
            static CONFIG: std::sync::OnceLock<AiConfig> = std::sync::OnceLock::new();
            CONFIG.get_or_init(|| {
                AiConfig::new(AiProvider::OpenAi)
                    .with_auth(AiAuth::BearerToken("test".to_string()))
                    .with_base_url("https://example.com")
                    .with_default_model("mock")
            })
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec!["mock".to_string()])
        }
    }

    #[derive(Clone)]
    struct EmptyAfterToolAiClient {
        calls: Arc<Mutex<Vec<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl AiClient for EmptyAfterToolAiClient {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
            self.calls.lock().unwrap().push(request.clone());
            let round = self.calls.lock().unwrap().len();

            match round {
                1 => Ok(ChatResponse {
                    id: "first".to_string(),
                    content: String::new(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    images: Vec::new(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "mcp_calc_sum".to_string(),
                        arguments: json!({"a": 2, "b": 3}),
                    }],
                    debug: None,
                }),
                2 => Ok(ChatResponse {
                    id: "empty-after-tool".to_string(),
                    content: String::new(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                    debug: None,
                }),
                _ => Ok(ChatResponse {
                    id: "final".to_string(),
                    content: "The tool result says the sum is 5.".to_string(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                    debug: None,
                }),
            }
        }

        fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
            stream::empty().boxed()
        }

        fn config(&self) -> &AiConfig {
            static CONFIG: std::sync::OnceLock<AiConfig> = std::sync::OnceLock::new();
            CONFIG.get_or_init(|| {
                AiConfig::new(AiProvider::OpenAi)
                    .with_auth(AiAuth::BearerToken("test".to_string()))
                    .with_base_url("https://example.com")
                    .with_default_model("mock")
            })
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec!["mock".to_string()])
        }
    }

    struct TestToolServer;

    impl ServerHandler for TestToolServer {
        async fn list_tools(
            &self,
            _request: Option<rmcp::model::PaginatedRequestParams>,
            _context: rmcp::service::RequestContext<rmcp::RoleServer>,
        ) -> Result<ListToolsResult, rmcp::ErrorData> {
            let mut schema = Map::new();
            schema.insert("type".to_string(), Value::String("object".to_string()));
            schema.insert(
                "properties".to_string(),
                json!({
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                }),
            );
            let mut tool = Tool::default();
            tool.name = "sum".into();
            tool.description = Some("Adds two integers".into());
            tool.input_schema = std::sync::Arc::new(schema);
            Ok(ListToolsResult {
                tools: vec![tool],
                next_cursor: None,
                meta: None,
            })
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: rmcp::service::RequestContext<rmcp::RoleServer>,
        ) -> Result<CallToolResult, rmcp::ErrorData> {
            let arguments = request.arguments.unwrap_or_default();
            let a = arguments
                .get("a")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let b = arguments
                .get("b")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            Ok(CallToolResult::structured(json!({ "sum": a + b })))
        }
    }

    #[test]
    fn parses_mcp_config_json() {
        let config = McpConfig::from_json_str(
            r#"{
                "mcpServers": {
                    "calc": {
                        "command": "uvx",
                        "args": ["calc-mcp"],
                        "env": { "MODE": "test" }
                    },
                    "remote": {
                        "url": "https://example.com/mcp",
                        "headers": { "x-api-key": "secret" }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.mcp_servers.len(), 2);
        assert_eq!(
            config
                .mcp_servers
                .get("calc")
                .and_then(|server| server.command.as_deref()),
            Some("uvx")
        );
        assert_eq!(
            config
                .mcp_servers
                .get("remote")
                .and_then(|server| server.url.as_deref()),
            Some("https://example.com/mcp")
        );
    }

    #[test]
    fn parses_claude_compatible_remote_fields() {
        let config = McpConfig::from_json_str(
            r#"{
                "mcpServers": {
                    "remote": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "headers": { "Authorization": "Bearer token" },
                        "headersHelper": "./scripts/headers-helper.cmd",
                        "oauth": {
                            "clientId": "client-123",
                            "callbackPort": 8080,
                            "authServerMetadataUrl": "https://auth.example.com/.well-known/openid-configuration"
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        let remote = config.mcp_servers.get("remote").unwrap();
        assert_eq!(remote.server_type.as_deref(), Some("http"));
        assert_eq!(
            remote.headers_helper.as_deref(),
            Some("./scripts/headers-helper.cmd")
        );
        let oauth = remote.oauth.as_ref().unwrap();
        assert_eq!(oauth.client_id.as_deref(), Some("client-123"));
        assert_eq!(oauth.callback_port, Some(8080));
        assert_eq!(
            oauth.auth_server_metadata_url.as_deref(),
            Some("https://auth.example.com/.well-known/openid-configuration")
        );
    }

    #[test]
    fn expands_env_vars_and_resolves_relative_paths_from_mcp_json() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base_dir = std::env::temp_dir().join(format!("connect-llm-mcp-{}", unique));
        fs::create_dir_all(base_dir.join("scripts")).unwrap();
        let config_path = base_dir.join("mcp.json");

        unsafe {
            std::env::set_var("MCP_API_TOKEN", "secret-token");
        }
        let json = r#"{
            "mcpServers": {
                "remote": {
                    "type": "http",
                    "url": "${API_BASE_URL:-https://api.example.com}/mcp",
                    "headers": {
                        "Authorization": "Bearer ${MCP_API_TOKEN}"
                    },
                    "headersHelper": "./scripts/helper.cmd"
                }
            }
        }"#;
        fs::write(&config_path, json).unwrap();

        let config = McpConfig::from_path(&config_path).unwrap();
        let remote = config.mcp_servers.get("remote").unwrap();
        assert_eq!(remote.url.as_deref(), Some("https://api.example.com/mcp"));
        assert_eq!(
            remote.headers.get("Authorization").map(String::as_str),
            Some("Bearer secret-token")
        );
        assert_eq!(
            remote.headers_helper.as_ref().map(PathBuf::from).as_deref(),
            Some(base_dir.join("scripts").join("helper.cmd").as_path())
        );

        let _ = fs::remove_file(&config_path);
        let _ = fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn headers_helper_output_is_merged_with_static_headers() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base_dir = std::env::temp_dir().join(format!("connect-llm-helper-{}", unique));
        fs::create_dir_all(&base_dir).unwrap();
        let helper_path = if cfg!(windows) {
            let path = base_dir.join("headers-helper.cmd");
            fs::write(
                &path,
                "@echo {\"Authorization\":\"Bearer dynamic\",\"X-Dynamic\":\"2\"}\r\n",
            )
            .unwrap();
            path
        } else {
            let path = base_dir.join("headers-helper.sh");
            fs::write(
                &path,
                "#!/bin/sh\nprintf '%s' '{\"Authorization\":\"Bearer dynamic\",\"X-Dynamic\":\"2\"}'\n",
            )
            .unwrap();
            path
        };
        let mut config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some("https://example.com/mcp".to_string()),
            headers: HashMap::from([
                ("Authorization".to_string(), "Bearer static".to_string()),
                ("X-Static".to_string(), "1".to_string()),
            ])
            .into_iter()
            .collect(),
            headers_helper: Some(helper_path.to_string_lossy().into_owned()),
            ..Default::default()
        };
        config.expand_env_vars("remote").unwrap();

        let merged = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(super::build_remote_headers(&config, Some("remote")))
            .unwrap();

        assert_eq!(
            merged.get("Authorization").map(String::as_str),
            Some("Bearer dynamic")
        );
        assert_eq!(merged.get("X-Static").map(String::as_str), Some("1"));
        assert_eq!(merged.get("X-Dynamic").map(String::as_str), Some("2"));

        let _ = fs::remove_file(&helper_path);
        let _ = fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn auth_header_is_applied_to_remote_requests() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let request_text = Arc::new(Mutex::new(String::new()));
        let request_text_clone = request_text.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 4096];
            let read_len = stream.read(&mut buffer).unwrap();
            *request_text_clone.lock().unwrap() =
                String::from_utf8_lossy(&buffer[..read_len]).to_string();
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some(format!("http://127.0.0.1:{}/mcp", port)),
            auth_header: Some("Bearer configured-token".to_string()),
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let headers = runtime
            .block_on(super::build_remote_headers(&config, Some("remote")))
            .unwrap();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer configured-token")
        );
        let client = super::build_remote_reqwest_client(&headers).unwrap();
        runtime.block_on(async {
            client
                .get(config.url.as_ref().unwrap())
                .send()
                .await
                .unwrap();
        });

        server.join().unwrap();
        assert!(
            request_text
                .lock()
                .unwrap()
                .to_ascii_lowercase()
                .contains("authorization: bearer configured-token"),
        );
    }

    #[test]
    fn remote_oauth_context_requires_explicit_oauth_config() {
        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some("https://example.com/mcp".to_string()),
            ..Default::default()
        };

        let context = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(super::RemoteOAuthContext::new(
                "remote",
                &config,
                reqwest::Client::new(),
            ))
            .unwrap();

        assert!(context.is_none());
    }

    #[test]
    fn oauth_metadata_source_preserves_configured_metadata_mode() {
        assert_eq!(
            super::oauth_metadata_source(&super::McpOAuthConfig {
                auth_server_metadata_url: Some(
                    "https://auth.example.com/.well-known/openid-configuration".to_string(),
                ),
                ..Default::default()
            }),
            super::OAuthMetadataSource::Configured
        );
        assert_eq!(
            super::oauth_metadata_source(&super::McpOAuthConfig::default()),
            super::OAuthMetadataSource::Discover
        );
    }

    #[test]
    fn oauth_callback_timeout_releases_listener_port() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();

        let error =
            super::wait_for_oauth_callback(listener, Duration::from_millis(50)).unwrap_err();
        assert!(error.to_string().contains("Timed out"));

        let rebound = std::net::TcpListener::bind(("127.0.0.1", port)).unwrap();
        drop(rebound);
    }

    #[test]
    fn claude_sse_transport_is_detected() {
        let config = super::McpServerConfig {
            server_type: Some("sse".to_string()),
            url: Some("https://example.com/sse".to_string()),
            ..Default::default()
        };

        let transport = super::detect_transport(&config).unwrap();
        assert!(matches!(transport, super::McpTransport::LegacySse));
    }

    #[test]
    fn claude_oauth_config_is_accepted_except_xaa() {
        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some("https://example.com/mcp".to_string()),
            oauth: Some(super::McpOAuthConfig {
                client_id: Some("client-123".to_string()),
                callback_port: Some(8080),
                auth_server_metadata_url: Some(
                    "https://auth.example.com/.well-known/openid-configuration".to_string(),
                ),
                xaa: Some(false),
            }),
            ..Default::default()
        };

        super::validate_supported_config(&config).unwrap();
    }

    #[test]
    fn claude_oauth_xaa_is_rejected_explicitly() {
        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some("https://example.com/mcp".to_string()),
            oauth: Some(super::McpOAuthConfig {
                client_id: None,
                callback_port: None,
                auth_server_metadata_url: None,
                xaa: Some(true),
            }),
            ..Default::default()
        };

        let error = super::validate_supported_config(&config).unwrap_err();
        assert!(error.to_string().contains("xaa"));
    }

    #[test]
    fn accepts_supported_negotiated_streamable_http_versions() {
        super::validate_negotiated_protocol_version(
            "2025-11-25",
            super::RemoteProtocolFlavor::StreamableHttp,
        )
        .unwrap();
        super::validate_negotiated_protocol_version(
            "2025-03-26",
            super::RemoteProtocolFlavor::StreamableHttp,
        )
        .unwrap();
    }

    #[test]
    fn rejects_unknown_negotiated_streamable_http_versions() {
        let error = super::validate_negotiated_protocol_version(
            "2099-01-01",
            super::RemoteProtocolFlavor::StreamableHttp,
        )
        .unwrap_err();
        assert!(error.to_string().contains("2099-01-01"));
    }

    #[test]
    fn repairs_incomplete_json_object_tool_arguments() {
        let object = super::value_as_json_object(&Value::String(
            "{\n  \"query\": \"さくらのAI Engine\",\n  \"max_results\": 10,\n  \"region\": \"\"\n"
                .to_string(),
        ))
        .unwrap();

        assert_eq!(
            object.get("query"),
            Some(&Value::String("さくらのAI Engine".to_string()))
        );
        assert_eq!(object.get("max_results"), Some(&json!(10)));
        assert_eq!(object.get("region"), Some(&Value::String(String::new())));
    }

    #[test]
    fn parses_json_like_text_tool_result_values() {
        assert_eq!(super::parse_json_like_text_value("[]"), Some(json!([])));
        assert_eq!(
            super::parse_json_like_text_value("{\"items\":[1,2]}"),
            Some(json!({ "items": [1, 2] }))
        );
        assert_eq!(super::parse_json_like_text_value("not json"), None);
    }

    #[tokio::test]
    async fn bridge_executes_mcp_tool_calls() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let mut sessions = super::McpSessionSet::from_pending(
            vec![
                PendingMcpServer::from_service(
                    "calc".to_string(),
                    Some("Calculator".to_string()),
                    service,
                )
                .await
                .unwrap(),
            ],
            HashMap::new(),
        );
        let bridge = McpBridge::new(McpConfig::default())
            .with_tool_loop_config(McpToolLoopConfig { max_round_trips: 4 });
        let client = MockAiClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        };

        let result = bridge
            .chat_with_sessions(
                None,
                &client,
                ChatRequest::new("mock", vec![Message::user("what is 2 + 3?")]),
                &mut sessions,
            )
            .await
            .unwrap();

        assert_eq!(result.tool_executions.len(), 1);
        assert_eq!(result.tool_executions[0].server_label, "calc");
        assert_eq!(result.tool_executions[0].remote_tool_name, "sum");
        assert!(result.response.content.contains("{\"sum\":5}"));

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(
            calls[0]
                .tools
                .iter()
                .any(|tool| tool.name == "mcp_calc_sum")
        );
        assert_eq!(
            calls[1]
                .messages
                .last()
                .and_then(|message| message.tool_result_value().cloned()),
            Some(json!({ "sum": 5 }))
        );

        sessions.close().await;
    }

    #[tokio::test]
    async fn bridge_falls_back_to_final_answer_when_tool_loop_limit_is_reached() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let mut sessions = super::McpSessionSet::from_pending(
            vec![
                PendingMcpServer::from_service(
                    "calc".to_string(),
                    Some("Calculator".to_string()),
                    service,
                )
                .await
                .unwrap(),
            ],
            HashMap::new(),
        );
        let bridge = McpBridge::new(McpConfig::default())
            .with_tool_loop_config(McpToolLoopConfig { max_round_trips: 2 });
        let client = LoopingAiClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        };

        let result = bridge
            .chat_with_sessions(
                None,
                &client,
                ChatRequest::new("mock", vec![Message::user("search for something")]),
                &mut sessions,
            )
            .await
            .unwrap();

        assert_eq!(
            result.response.content,
            "I could not get search results from the tool."
        );
        assert!(result.response.tool_calls.is_empty());
        assert!(
            result
                .messages
                .iter()
                .any(|message| message.role() == "tool")
        );
        assert_eq!(client.calls.lock().unwrap().len(), 4);

        sessions.close().await;
    }

    #[tokio::test]
    async fn bridge_requests_final_visible_answer_after_empty_post_tool_response() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let mut sessions = super::McpSessionSet::from_pending(
            vec![
                PendingMcpServer::from_service(
                    "calc".to_string(),
                    Some("Calculator".to_string()),
                    service,
                )
                .await
                .unwrap(),
            ],
            HashMap::new(),
        );
        let bridge = McpBridge::new(McpConfig::default())
            .with_tool_loop_config(McpToolLoopConfig { max_round_trips: 4 });
        let client = EmptyAfterToolAiClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        };

        let result = bridge
            .chat_with_sessions(
                None,
                &client,
                ChatRequest::new("mock", vec![Message::user("what is 2 + 3?")]),
                &mut sessions,
            )
            .await
            .unwrap();

        assert_eq!(
            result.response.content,
            "The tool result says the sum is 5."
        );
        assert_eq!(client.calls.lock().unwrap().len(), 3);

        sessions.close().await;
    }

    #[tokio::test]
    async fn session_status_reports_connected_servers_and_tools() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let sessions = super::McpSessionSet::from_pending(
            vec![
                PendingMcpServer::from_service(
                    "calc".to_string(),
                    Some("Calculator".to_string()),
                    service,
                )
                .await
                .unwrap(),
            ],
            HashMap::new(),
        );

        let mut config = McpConfig::default();
        config.mcp_servers.insert(
            "calc".to_string(),
            super::McpServerConfig {
                command: Some("uvx".to_string()),
                args: vec!["calc-mcp".to_string()],
                ..Default::default()
            },
        );

        let status = sessions.status_with_config(&config);
        assert!(status.connected);
        assert_eq!(status.connected_server_count, 1);
        assert_eq!(status.configured_server_count, 1);
        assert_eq!(status.configured_servers[0].label, "calc");
        assert!(status.configured_servers[0].connected);
        assert!(status.configured_servers[0].last_error.is_none());
        assert_eq!(status.exported_tools.len(), 1);
        assert_eq!(status.exported_tools[0].alias, "mcp_calc_sum");
        assert_eq!(status.exported_tools[0].remote_tool_name, "sum");

        sessions.close().await;
    }

    #[test]
    fn bridge_status_reports_connection_errors_without_failing_runtime_status() {
        let mut config = McpConfig::default();
        config.mcp_servers.insert(
            "remote".to_string(),
            super::McpServerConfig {
                url: Some("https://example.com/mcp".to_string()),
                ..Default::default()
            },
        );

        let sessions = super::McpSessionSet::from_pending(
            Vec::new(),
            HashMap::from([(
                "remote".to_string(),
                "API error: MCP error: Transport channel closed".to_string(),
            )]),
        );
        let status = sessions.status_with_config(&config);

        assert!(!status.connected);
        assert_eq!(status.configured_server_count, 1);
        assert_eq!(status.connected_server_count, 0);
        assert_eq!(
            status.configured_servers[0].last_error.as_deref(),
            Some("API error: MCP error: Transport channel closed")
        );
    }

    #[derive(Clone)]
    struct MockStreamingAiClient {
        calls: Arc<Mutex<Vec<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl AiClient for MockStreamingAiClient {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, AiError> {
            Err(AiError::api(
                "MockStreamingAiClient::chat should not be used in this test".to_string(),
            ))
        }

        fn chat_stream(
            &self,
            request: ChatRequest,
        ) -> BoxStream<'static, Result<StreamChunk, AiError>> {
            self.calls.lock().unwrap().push(request.clone());
            let round = self.calls.lock().unwrap().len();
            match round {
                1 => stream::iter(vec![
                    Ok(StreamChunk {
                        delta: "Let me calculate that. ".to_string(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: Vec::new(),
                        done: false,
                        debug: None,
                    }),
                    Ok(StreamChunk {
                        delta: String::new(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: vec![ToolCallDelta {
                            index: 0,
                            id: Some("call_1".to_string()),
                            name: Some("mcp_calc_sum".to_string()),
                            arguments: Some("{\"a\":2".to_string()),
                        }],
                        done: false,
                        debug: None,
                    }),
                    Ok(StreamChunk {
                        delta: String::new(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: vec![ToolCallDelta {
                            index: 0,
                            id: None,
                            name: None,
                            arguments: Some(",\"b\":3}".to_string()),
                        }],
                        done: true,
                        debug: None,
                    }),
                ])
                .boxed(),
                _ => stream::iter(vec![
                    Ok(StreamChunk {
                        delta: "The answer is 5.".to_string(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: Vec::new(),
                        done: false,
                        debug: None,
                    }),
                    Ok(StreamChunk {
                        delta: String::new(),
                        thinking_delta: None,
                        thinking_signature: None,
                        images: Vec::new(),
                        tool_call_deltas: Vec::new(),
                        done: true,
                        debug: None,
                    }),
                ])
                .boxed(),
            }
        }

        fn config(&self) -> &AiConfig {
            static CONFIG: std::sync::OnceLock<AiConfig> = std::sync::OnceLock::new();
            CONFIG.get_or_init(|| {
                AiConfig::new(AiProvider::OpenAi)
                    .with_auth(AiAuth::BearerToken("test".to_string()))
                    .with_base_url("https://example.com")
                    .with_default_model("mock")
            })
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec!["mock".to_string()])
        }
    }

    #[tokio::test]
    async fn bridge_stream_executes_mcp_tool_calls() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let mut sessions = super::McpSessionSet::from_pending(
            vec![
                PendingMcpServer::from_service(
                    "calc".to_string(),
                    Some("Calculator".to_string()),
                    service,
                )
                .await
                .unwrap(),
            ],
            HashMap::new(),
        );
        let bridge = McpBridge::new(McpConfig::default())
            .with_tool_loop_config(McpToolLoopConfig { max_round_trips: 4 });
        let client = MockStreamingAiClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        };

        let mut stream = bridge.chat_stream_with_sessions(
            None,
            &client,
            ChatRequest::new("mock", vec![Message::user("what is 2 + 3?")]),
            &mut sessions,
        );

        let mut rendered = String::new();
        let mut final_response = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                super::McpStreamEvent::Chunk(chunk) => rendered.push_str(&chunk.delta),
                super::McpStreamEvent::Finished(managed) => final_response = Some(managed),
            }
        }
        drop(stream);

        let managed = final_response.expect("missing final stream response");
        assert_eq!(rendered, "The answer is 5.");
        assert_eq!(managed.tool_executions.len(), 1);
        assert_eq!(managed.tool_executions[0].remote_tool_name, "sum");
        assert_eq!(managed.response.content, "The answer is 5.");
        assert_eq!(
            managed
                .messages
                .iter()
                .filter(|message| message.role() == "tool")
                .count(),
            1
        );

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1]
                .messages
                .last()
                .and_then(|message| message.tool_result_value().cloned()),
            Some(json!({ "sum": 5 }))
        );

        sessions.close().await;
    }

    #[tokio::test]
    async fn legacy_sse_timeout_removes_pending_entry() {
        let pending = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        let (sender, receiver) = tokio::sync::oneshot::channel();
        pending.lock().await.insert("1".to_string(), sender);

        let error = super::await_pending_legacy_sse_response(
            &pending,
            "1",
            "tools/list",
            receiver,
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("Timed out"));
        assert!(pending.lock().await.is_empty());
    }

    #[tokio::test]
    async fn legacy_sse_send_request_removes_pending_entry_when_post_fails() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let client = super::LegacySseClient {
            base_url: format!("http://127.0.0.1:{port}/sse"),
            client: reqwest::Client::new(),
            oauth: None,
            endpoint_url: tokio::sync::Mutex::new(format!("http://127.0.0.1:{port}/messages")),
            protocol_version: tokio::sync::Mutex::new(
                super::MCP_PROTOCOL_VERSION_LEGACY_SSE.to_string(),
            ),
            next_request_id: std::sync::atomic::AtomicU64::new(0),
            tool_state: Arc::new(super::RemoteToolCatalogState::default()),
            pending: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            shutdown_tx: tokio::sync::Mutex::new(None),
            reader_task: tokio::sync::Mutex::new(None),
        };

        let error = client
            .send_request_value("tools/list", Some(json!({})))
            .await
            .unwrap_err();

        assert!(
            matches!(
                error.kind,
                crate::ai::AiErrorKind::Http | crate::ai::AiErrorKind::Api
            ),
            "unexpected error: {error}"
        );
        assert!(client.pending.lock().await.is_empty());
    }

    #[tokio::test]
    async fn legacy_sse_close_clears_pending_entries() {
        let pending = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        let (sender, receiver) = tokio::sync::oneshot::channel();
        pending.lock().await.insert("1".to_string(), sender);

        let mut client = super::LegacySseClient {
            base_url: "http://127.0.0.1:1/sse".to_string(),
            client: reqwest::Client::new(),
            oauth: None,
            endpoint_url: tokio::sync::Mutex::new("http://127.0.0.1:1/messages".to_string()),
            protocol_version: tokio::sync::Mutex::new(
                super::MCP_PROTOCOL_VERSION_LEGACY_SSE.to_string(),
            ),
            next_request_id: std::sync::atomic::AtomicU64::new(0),
            tool_state: Arc::new(super::RemoteToolCatalogState::default()),
            pending: pending.clone(),
            shutdown_tx: tokio::sync::Mutex::new(None),
            reader_task: tokio::sync::Mutex::new(None),
        };

        client.close().await.unwrap();

        assert!(pending.lock().await.is_empty());
        assert!(receiver.await.is_err());
    }

    #[tokio::test]
    async fn streamable_http_reinitializes_after_404_and_deletes_session_on_close() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut seen_requests = Vec::new();
            let mut step = 0;
            while step < 7 {
                let (mut stream, _) = listener.accept().unwrap();
                let (request_line, headers, body) = read_http_request(&mut stream);
                seen_requests.push((request_line.clone(), headers.clone()));
                let method = request_line.split_whitespace().next().unwrap_or_default();
                if method == "GET" {
                    write_http_response(&mut stream, "405 Method Not Allowed", &[], "");
                    continue;
                }
                let request_json = if body.trim().is_empty() {
                    None
                } else {
                    Some(serde_json::from_str::<Value>(&body).unwrap())
                };
                match step {
                    0 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("initialize")
                        );
                        assert!(!headers.contains_key("mcp-session-id"));
                        write_http_response(
                            &mut stream,
                            "200 OK",
                            &[
                                ("Content-Type", "application/json"),
                                ("MCP-Session-Id", "session-1"),
                            ],
                            &json!({
                                "jsonrpc": "2.0",
                                "id": request_json.as_ref().and_then(|value| value.get("id")).cloned().unwrap(),
                                "result": {
                                    "protocolVersion": "2025-11-25",
                                    "capabilities": {},
                                    "serverInfo": { "name": "test", "version": "1.0.0" }
                                }
                            })
                            .to_string(),
                        );
                    }
                    1 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("notifications/initialized")
                        );
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-1")
                        );
                        write_http_response(&mut stream, "202 Accepted", &[], "");
                    }
                    2 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("tools/list")
                        );
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-1")
                        );
                        write_http_response(&mut stream, "404 Not Found", &[], "");
                    }
                    3 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("initialize")
                        );
                        assert!(!headers.contains_key("mcp-session-id"));
                        write_http_response(
                            &mut stream,
                            "200 OK",
                            &[
                                ("Content-Type", "application/json"),
                                ("MCP-Session-Id", "session-2"),
                            ],
                            &json!({
                                "jsonrpc": "2.0",
                                "id": request_json.as_ref().and_then(|value| value.get("id")).cloned().unwrap(),
                                "result": {
                                    "protocolVersion": "2025-11-25",
                                    "capabilities": {},
                                    "serverInfo": { "name": "test", "version": "1.0.0" }
                                }
                            })
                            .to_string(),
                        );
                    }
                    4 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("notifications/initialized")
                        );
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-2")
                        );
                        write_http_response(&mut stream, "202 Accepted", &[], "");
                    }
                    5 => {
                        assert_eq!(method, "POST");
                        assert_eq!(
                            request_json
                                .as_ref()
                                .and_then(|value| value.get("method"))
                                .and_then(Value::as_str),
                            Some("tools/list")
                        );
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-2")
                        );
                        write_http_response(
                            &mut stream,
                            "200 OK",
                            &[("Content-Type", "application/json")],
                            &json!({
                                "jsonrpc": "2.0",
                                "id": request_json.as_ref().and_then(|value| value.get("id")).cloned().unwrap(),
                                "result": {
                                    "tools": [{
                                        "name": "search",
                                        "description": "Search",
                                        "inputSchema": { "type": "object" }
                                    }],
                                    "nextCursor": null
                                }
                            })
                            .to_string(),
                        );
                    }
                    6 => {
                        assert_eq!(method, "DELETE");
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-2")
                        );
                        write_http_response(&mut stream, "204 No Content", &[], "");
                    }
                    _ => unreachable!(),
                }
                step += 1;
            }
            seen_requests
        });

        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some(format!("http://{}/mcp", address)),
            ..Default::default()
        };
        let client =
            super::RemoteHttpClient::connect("remote", &config, reqwest::Client::new(), None)
                .await
                .unwrap();
        let tools = client.list_all_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name.as_ref(), "search");
        client.close().await.unwrap();

        let seen_requests = server.join().unwrap();
        assert_eq!(
            seen_requests
                .iter()
                .filter(|(request_line, _)| !request_line.starts_with("GET "))
                .count(),
            7
        );
    }

    #[tokio::test]
    async fn streamable_http_listener_answers_ping_and_refreshes_tools_after_list_changed() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut saw_initialize = false;
            let mut saw_initialized = false;
            let mut saw_get = false;
            let mut saw_ping_response = false;
            let mut saw_tools_list = false;
            let mut saw_delete = false;

            while !(saw_initialize
                && saw_initialized
                && saw_get
                && saw_ping_response
                && saw_tools_list
                && saw_delete)
            {
                let (mut stream, _) = listener.accept().unwrap();
                let (request_line, headers, body) = read_http_request(&mut stream);
                let method = request_line.split_whitespace().next().unwrap_or_default();
                if method == "POST" {
                    let request_json = serde_json::from_str::<Value>(&body).unwrap();
                    match request_json.get("method").and_then(Value::as_str) {
                        Some("initialize") => {
                            assert!(!saw_initialize);
                            saw_initialize = true;
                            write_http_response(
                                &mut stream,
                                "200 OK",
                                &[
                                    ("Content-Type", "application/json"),
                                    ("MCP-Session-Id", "session-1"),
                                ],
                                &json!({
                                    "jsonrpc": "2.0",
                                    "id": request_json.get("id").cloned().unwrap(),
                                    "result": {
                                        "protocolVersion": "2025-11-25",
                                        "capabilities": {
                                            "tools": { "listChanged": true }
                                        },
                                        "serverInfo": { "name": "test", "version": "1.0.0" }
                                    }
                                })
                                .to_string(),
                            );
                        }
                        Some("notifications/initialized") => {
                            assert!(saw_initialize);
                            saw_initialized = true;
                            write_http_response(&mut stream, "202 Accepted", &[], "");
                        }
                        Some("tools/list") => {
                            saw_tools_list = true;
                            write_http_response(
                                &mut stream,
                                "200 OK",
                                &[("Content-Type", "application/json")],
                                &json!({
                                    "jsonrpc": "2.0",
                                    "id": request_json.get("id").cloned().unwrap(),
                                    "result": {
                                        "tools": [{
                                            "name": "search",
                                            "description": "Search",
                                            "inputSchema": { "type": "object" }
                                        },{
                                            "name": "fetch",
                                            "description": "Fetch",
                                            "inputSchema": { "type": "object" }
                                        }],
                                        "nextCursor": null
                                    }
                                })
                                .to_string(),
                            );
                        }
                        _ => {
                            assert_eq!(
                                request_json.get("id").and_then(Value::as_str),
                                Some("server-1")
                            );
                            assert_eq!(request_json.get("result"), Some(&json!({})));
                            saw_ping_response = true;
                            write_http_response(&mut stream, "202 Accepted", &[], "");
                        }
                    }
                    continue;
                }

                if method == "GET" {
                    assert_eq!(
                        headers.get("mcp-session-id").map(String::as_str),
                        Some("session-1")
                    );
                    saw_get = true;
                    let body = concat!(
                        "id: evt-1\r\n",
                        "data: {\"jsonrpc\":\"2.0\",\"id\":\"server-1\",\"method\":\"ping\"}\r\n\r\n",
                        "id: evt-2\r\n",
                        "retry: 25\r\n",
                        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\r\n\r\n",
                    );
                    write_http_response(
                        &mut stream,
                        "200 OK",
                        &[("Content-Type", "text/event-stream")],
                        body,
                    );
                    continue;
                }

                assert_eq!(method, "DELETE");
                saw_delete = true;
                write_http_response(&mut stream, "204 No Content", &[], "");
            }
        });

        let config = super::McpServerConfig {
            server_type: Some("http".to_string()),
            url: Some(format!("http://{}/mcp", address)),
            ..Default::default()
        };
        let client =
            super::RemoteHttpClient::connect("remote", &config, reqwest::Client::new(), None)
                .await
                .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let refreshed = client
            .refresh_tools_if_needed()
            .await
            .unwrap()
            .expect("expected tool refresh after list_changed");
        assert_eq!(refreshed.len(), 2);
        assert_eq!(refreshed[0].name.as_ref(), "search");
        assert_eq!(refreshed[1].name.as_ref(), "fetch");
        let last_event_id = client.tool_state.last_event_id.lock().await.clone();
        assert!(matches!(
            last_event_id.as_deref(),
            Some("evt-1") | Some("evt-2")
        ));
        client.close().await.unwrap();
        server.join().unwrap();
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_browser_launcher_uses_direct_url_argument() {
        let url = "https://example.com/oauth?code=abc123&state=xyz789";
        let command = super::browser_open_command_local(url);
        let args: Vec<&OsStr> = command.get_args().collect();

        assert_eq!(command.get_program(), OsStr::new("explorer"));
        assert_eq!(args, vec![OsStr::new(url)]);
    }

    #[allow(dead_code)]
    fn _assert_send(_: McpManagedChatResponse) {}

    fn read_http_request(
        stream: &mut std::net::TcpStream,
    ) -> (String, HashMap<String, String>, String) {
        use std::io::Read;

        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 4096];
        let header_end = loop {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0, "unexpected EOF while reading request headers");
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };

        let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let mut lines = header_text.split("\r\n").filter(|line| !line.is_empty());
        let request_line = lines.next().unwrap_or_default().to_string();
        let mut headers = HashMap::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut body_bytes = buffer[header_end..].to_vec();
        while body_bytes.len() < content_length {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0, "unexpected EOF while reading request body");
            body_bytes.extend_from_slice(&chunk[..read]);
        }
        body_bytes.truncate(content_length);
        let body = String::from_utf8_lossy(&body_bytes).to_string();
        (request_line, headers, body)
    }

    fn write_http_response(
        stream: &mut std::net::TcpStream,
        status: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) {
        use std::io::Write;

        let mut response = format!(
            "HTTP/1.1 {}\r\nConnection: close\r\nContent-Length: {}\r\n",
            status,
            body.len()
        );
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        response.push_str(body);
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    }
}
