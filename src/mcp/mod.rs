use crate::{
    ai::{
        AiClient, AiError, ChatRequest, ChatResponse, DebugTrace, GeneratedImage, Message,
        StreamChunk, ThinkingOutput, ToolCall, ToolCallDelta, ToolDefinition, Usage,
        debug_logging_enabled,
    },
    context::{ContextCompaction, ContextManager, ManagedChatResponse, PreparedChatRequest},
};
use async_stream::stream;
use futures_util::{StreamExt, stream, stream::BoxStream};
use http::{HeaderName, HeaderValue};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, Content, Tool},
    service::{Peer, RunningService},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
    path::PathBuf,
    process::Stdio,
    time::Duration,
};
use tokio::process::Command;
use tokio::time::timeout;

const MAX_IDENTICAL_TOOL_CALLS: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers", alias = "servers")]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
}

impl McpConfig {
    pub fn from_json_str(json: &str) -> Result<Self, AiError> {
        serde_json::from_str(json).map_err(|error| AiError::Parse(error.to_string()))
    }

    pub fn from_json_value(value: Value) -> Result<Self, AiError> {
        serde_json::from_value(value).map_err(|error| AiError::Parse(error.to_string()))
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, AiError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| AiError::Api(error.to_string()))?;
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
        let sessions = McpSessionSet::connect(&self.config).await?;
        let result = self
            .chat_with_sessions(context_manager, client, request, &sessions)
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
            let mut request = request;
            let mcp_tools = sessions.tool_definitions();
            if !mcp_tools.is_empty() {
                request.tools.extend(mcp_tools);
            }

            let mut tool_executions = Vec::new();
            let mut tool_call_counts = HashMap::<String, usize>::new();
            let mut pending_error = None;
            let mut finished = false;

            'rounds: for round in 0..=self.tool_loop.max_round_trips {
                let prepared = match prepare_stream_request(context_manager, client, request.clone()).await {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        pending_error = Some(error);
                        break 'rounds;
                    }
                };
                let mut provider_stream = client.chat_stream(prepared.request.clone());
                let mut response_builder = StreamResponseBuilder::new(prepared.request.model.clone());
                let mut round_chunks = Vec::new();

                while let Some(next) = provider_stream.next().await {
                    match next {
                        Ok(chunk) => {
                            response_builder.ingest(&chunk);
                            round_chunks.push(chunk.clone());
                            if chunk.done {
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
                        match collect_stream_response(
                            context_manager,
                            client,
                            build_final_no_tools_request(
                                request.clone(),
                                "You have already gathered tool results. Do not call more tools. Provide a direct user-visible answer based on the information already gathered.",
                            ),
                        )
                        .await
                        {
                            Ok((chunks, final_managed)) => {
                                for chunk in chunks {
                                    yield Ok(McpStreamEvent::Chunk(chunk));
                                }
                                let final_response = final_managed.response.clone();
                                yield Ok(McpStreamEvent::Finished(McpManagedChatResponse {
                                    response: final_response.clone(),
                                    compaction: final_managed.compaction.or(prepared.compaction.clone()),
                                    messages: final_managed_messages_from_request(
                                        final_response,
                                        &request,
                                    ),
                                    tool_executions,
                                }));
                                finished = true;
                                break 'rounds;
                            }
                            Err(error) => {
                                pending_error = Some(error);
                                break 'rounds;
                            }
                        }
                    }

                    for chunk in round_chunks {
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

                    match collect_stream_response(
                        context_manager,
                        client,
                        build_final_no_tools_request(request.clone(), final_reason),
                    )
                    .await
                    {
                        Ok((chunks, final_managed)) => {
                            for chunk in chunks {
                                yield Ok(McpStreamEvent::Chunk(chunk));
                            }
                            let final_response = final_managed.response.clone();
                            yield Ok(McpStreamEvent::Finished(McpManagedChatResponse {
                                response: final_response.clone(),
                                compaction: final_managed.compaction,
                                messages: final_managed_messages_from_request(
                                    final_response,
                                    &request,
                                ),
                                tool_executions,
                            }));
                            finished = true;
                            break 'rounds;
                        }
                        Err(error) => {
                            pending_error = Some(error);
                            break 'rounds;
                        }
                    }
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
                yield Err(AiError::Api(
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
        sessions: &McpSessionSet,
    ) -> Result<McpManagedChatResponse, AiError> {
        let mcp_tools = sessions.tool_definitions();
        if !mcp_tools.is_empty() {
            request.tools.extend(mcp_tools);
        }

        let mut tool_executions = Vec::new();
        let mut tool_call_counts = HashMap::<String, usize>::new();

        for round in 0..=self.tool_loop.max_round_trips {
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
            (_, None) => Err(AiError::Api(
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
            (_, None) => Err(AiError::Api(
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
                Err(AiError::Api(
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
                Err(AiError::Api(
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
    service: RunningService<RoleClient, ()>,
    peer: Peer<RoleClient>,
    tools: Vec<Tool>,
}

impl PendingMcpServer {
    async fn connect(server_label: &str, config: &McpServerConfig) -> Result<Self, AiError> {
        let service = connect_peer(config).await?;
        Self::from_service(
            server_label.to_string(),
            config.description.clone(),
            service,
        )
        .await
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
            service,
            peer,
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
        let mut used_aliases = HashSet::new();
        let mut servers = Vec::new();
        let mut exported_tools = Vec::new();
        let mut tool_index = HashMap::new();

        for pending_server in pending {
            let server_index = servers.len();
            let mut aliases = HashMap::new();
            for tool in pending_server.tools {
                let remote_tool_name = tool.name.to_string();
                let alias = allocate_tool_alias(
                    &pending_server.server_label,
                    &remote_tool_name,
                    &mut used_aliases,
                );
                let input_schema = Value::Object((*tool.input_schema).clone());
                let description = build_tool_description(
                    &pending_server.server_label,
                    pending_server.description.as_deref(),
                    &remote_tool_name,
                    tool.description.as_deref(),
                );

                exported_tools.push(ToolDefinition::function(
                    alias.clone(),
                    Some(description),
                    input_schema,
                ));
                aliases.insert(alias.clone(), remote_tool_name.clone());
                tool_index.insert(
                    alias.clone(),
                    ResolvedMcpTool {
                        server_index,
                        server_label: pending_server.server_label.clone(),
                    },
                );
            }

            servers.push(McpServerSession {
                server_label: pending_server.server_label,
                service: pending_server.service,
                peer: pending_server.peer,
                aliases,
            });
        }

        Self {
            servers,
            exported_tools,
            tool_index,
            connect_errors,
        }
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.exported_tools.clone()
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
            let _ = server.service.close().await;
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
                AiError::Api(format!(
                    "MCP bridge cannot execute unknown tool call '{}'",
                    tool_call.name
                ))
            })?;
            let server = self.servers.get(resolved.server_index).ok_or_else(|| {
                AiError::Api(format!(
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
                        AiError::Api(format!(
                            "MCP bridge cannot map tool '{}' back to server '{}'",
                            tool_call.name, resolved.server_label
                        ))
                    })?;
            let arguments = value_as_json_object(&tool_call.arguments)?;
            let result = server
                .peer
                .call_tool(
                    CallToolRequestParams::new(Cow::Owned(remote_tool_name.clone()))
                        .with_arguments(arguments),
                )
                .await
                .map_err(mcp_service_error)?;
            let result_value = call_tool_result_to_value(&result);
            let is_error = result.is_error.unwrap_or(false);
            let mut message = Message::tool_result(
                tool_call.id.clone(),
                tool_call.name.clone(),
                result_value.clone(),
            );
            if is_error {
                message.tool_error = Some(true);
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
}

struct McpServerSession {
    server_label: String,
    service: RunningService<RoleClient, ()>,
    peer: Peer<RoleClient>,
    aliases: HashMap<String, String>,
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

async fn collect_stream_response(
    context_manager: Option<&ContextManager>,
    client: &dyn AiClient,
    request: ChatRequest,
) -> Result<(Vec<StreamChunk>, ManagedChatResponse), AiError> {
    let prepared = prepare_stream_request(context_manager, client, request).await?;
    let mut provider_stream = client.chat_stream(prepared.request.clone());
    let mut response_builder = StreamResponseBuilder::new(prepared.request.model.clone());
    let mut chunks = Vec::new();

    while let Some(next) = provider_stream.next().await {
        let chunk = next?;
        response_builder.ingest(&chunk);
        chunks.push(chunk.clone());
        if chunk.done {
            break;
        }
    }

    Ok((
        chunks,
        ManagedChatResponse {
            response: response_builder.finish(),
            compaction: prepared.compaction,
        },
    ))
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
            .any(|message| message.role == "tool")
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
            message.tool_error = Some(true);
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
            message.tool_error = Some(true);
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

async fn connect_peer(config: &McpServerConfig) -> Result<RunningService<RoleClient, ()>, AiError> {
    validate_supported_config(config)?;
    match detect_transport(config)? {
        McpTransport::Stdio => {
            let command = config.command.as_ref().ok_or_else(|| {
                AiError::Parse("MCP stdio server is missing 'command'".to_string())
            })?;
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
                .map_err(|error| AiError::Api(error.to_string()))?;
            ().serve(transport).await.map_err(mcp_service_error)
        }
        McpTransport::StreamableHttp => {
            let url = config
                .url
                .as_ref()
                .ok_or_else(|| AiError::Parse("MCP HTTP server is missing 'url'".to_string()))?;
            let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.clone())
                .reinit_on_expired_session(true);
            if let Some(auth_header) = &config.auth_header {
                transport_config = transport_config.auth_header(auth_header.clone());
            }
            let combined_headers = build_remote_headers(config).await?;
            if !combined_headers.is_empty() {
                transport_config =
                    transport_config.custom_headers(build_custom_headers(&combined_headers)?);
            }
            let transport = StreamableHttpClientTransport::from_config(transport_config);
            ().serve(transport).await.map_err(mcp_service_error)
        }
    }
}

fn detect_transport(config: &McpServerConfig) -> Result<McpTransport, AiError> {
    if let Some(server_type) = config.server_type.as_deref() {
        return match server_type {
            "stdio" => Ok(McpTransport::Stdio),
            "http" | "streamable_http" => Ok(McpTransport::StreamableHttp),
            "sse" => Err(AiError::Parse(
                "Claude-compatible MCP type 'sse' is not supported by this build yet".to_string(),
            )),
            "ws" => Err(AiError::Parse(
                "Claude-compatible MCP type 'ws' is not supported by this build yet".to_string(),
            )),
            "sdk" => Err(AiError::Parse(
                "Claude-compatible MCP type 'sdk' is not supported by this build yet".to_string(),
            )),
            "claudeai-proxy" => Err(AiError::Parse(
                "Claude-compatible MCP type 'claudeai-proxy' is not supported by this build yet"
                    .to_string(),
            )),
            other => Err(AiError::Parse(format!(
                "Unsupported Claude-compatible MCP type '{}'",
                other
            ))),
        };
    }
    if let Some(transport) = config.transport.as_deref() {
        return match transport {
            "stdio" => Ok(McpTransport::Stdio),
            "http" | "streamable_http" => Ok(McpTransport::StreamableHttp),
            "sse" => Err(AiError::Parse(
                "MCP transport 'sse' is not supported by this build yet".to_string(),
            )),
            other => Err(AiError::Parse(format!(
                "Unsupported MCP transport '{}'",
                other
            ))),
        };
    }

    match (config.command.is_some(), config.url.is_some()) {
        (true, false) => Ok(McpTransport::Stdio),
        (false, true) => Ok(McpTransport::StreamableHttp),
        (true, true) => Err(AiError::Parse(
            "MCP server config cannot specify both 'command' and 'url' without an explicit transport"
                .to_string(),
        )),
        (false, false) => Err(AiError::Parse(
            "MCP server config must specify either 'command' or 'url'".to_string(),
        )),
    }
}

fn validate_supported_config(config: &McpServerConfig) -> Result<(), AiError> {
    if let Some(oauth) = &config.oauth {
        if oauth.client_id.is_some()
            || oauth.callback_port.is_some()
            || oauth.auth_server_metadata_url.is_some()
            || oauth.xaa.unwrap_or(false)
        {
            return Err(AiError::Parse(
                "Claude-compatible MCP OAuth config is not supported by this build yet".to_string(),
            ));
        }
    }
    Ok(())
}

async fn build_remote_headers(
    config: &McpServerConfig,
) -> Result<BTreeMap<String, String>, AiError> {
    let mut headers = config.headers.clone();
    if let Some(dynamic_headers) = run_headers_helper(config).await? {
        headers.extend(dynamic_headers);
    }
    Ok(headers)
}

async fn run_headers_helper(
    config: &McpServerConfig,
) -> Result<Option<BTreeMap<String, String>>, AiError> {
    let Some(helper) = config.headers_helper.as_deref() else {
        return Ok(None);
    };

    let mut command = shell_command_for_platform(helper);
    let output = timeout(Duration::from_secs(10), command.output())
        .await
        .map_err(|_| AiError::Api("MCP headersHelper timed out after 10 seconds".to_string()))?
        .map_err(|error| AiError::Api(format!("Failed to execute MCP headersHelper: {}", error)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit code {:?}", output.status.code())
        } else {
            stderr
        };
        return Err(AiError::Api(format!(
            "MCP headersHelper failed: {}",
            detail
        )));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        AiError::Parse(format!(
            "MCP headersHelper returned non-UTF-8 output: {}",
            error
        ))
    })?;
    let parsed: Value = serde_json::from_str(stdout.trim()).map_err(|error| {
        AiError::Parse(format!(
            "MCP headersHelper must return a JSON object: {}",
            error
        ))
    })?;
    let object = parsed
        .as_object()
        .ok_or_else(|| AiError::Parse("MCP headersHelper must return a JSON object".to_string()))?;

    let mut headers = BTreeMap::new();
    for (key, value) in object {
        let value = value.as_str().ok_or_else(|| {
            AiError::Parse(format!(
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

fn build_custom_headers(
    headers: &BTreeMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>, AiError> {
    let mut parsed = HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = name
            .parse()
            .map_err(|error| AiError::Parse(format!("Invalid MCP header '{}': {}", name, error)))?;
        let header_value = HeaderValue::from_str(value).map_err(|error| {
            AiError::Parse(format!(
                "Invalid MCP header value for '{}': {}",
                name, error
            ))
        })?;
        parsed.insert(header_name, header_value);
    }
    Ok(parsed)
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
            AiError::Parse(format!(
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
        Err(AiError::Parse(format!(
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
    let mut message = Message::assistant(response.content.clone());
    message.thinking = response.thinking.clone();
    message.tool_calls = response.tool_calls.clone();
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
            AiError::Parse(format!(
                "MCP tool arguments must be a JSON object, got {}",
                value
            ))
        }),
        other => Err(AiError::Parse(format!(
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
            return Value::String(text.text.clone());
        }
    }

    json!({
        "content": result.content.iter().map(content_to_value).collect::<Vec<_>>(),
        "is_error": result.is_error.unwrap_or(false),
    })
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
    AiError::Api(format!("MCP error: {}", error))
}

#[derive(Debug)]
enum McpTransport {
    Stdio,
    StreamableHttp,
}

#[cfg(test)]
mod tests {
    use super::{
        McpBridge, McpConfig, McpManagedChatResponse, McpToolLoopConfig, PendingMcpServer,
    };
    use crate::ai::{
        AiClient, AiConfig, AiError, ChatRequest, ChatResponse, Message, StreamChunk, ToolCall,
        ToolCallDelta, Usage,
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
        fs,
        path::PathBuf,
        sync::{Arc, Mutex},
        time::{SystemTime, UNIX_EPOCH},
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
                        .and_then(|message| message.tool_result.clone())
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
            CONFIG.get_or_init(|| AiConfig {
                api_key: "test".to_string(),
                base_url: "https://example.com".to_string(),
                model: "mock".to_string(),
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
            CONFIG.get_or_init(|| AiConfig {
                api_key: "test".to_string(),
                base_url: "https://example.com".to_string(),
                model: "mock".to_string(),
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
            CONFIG.get_or_init(|| AiConfig {
                api_key: "test".to_string(),
                base_url: "https://example.com".to_string(),
                model: "mock".to_string(),
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
            .block_on(super::build_remote_headers(&config))
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
    fn claude_sse_transport_is_rejected_explicitly() {
        let config = super::McpServerConfig {
            server_type: Some("sse".to_string()),
            url: Some("https://example.com/sse".to_string()),
            ..Default::default()
        };

        let error = super::detect_transport(&config).unwrap_err();
        assert!(error.to_string().contains("type 'sse'"));
    }

    #[test]
    fn claude_oauth_config_is_rejected_explicitly() {
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

        let error = super::validate_supported_config(&config).unwrap_err();
        assert!(error.to_string().contains("OAuth config"));
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

    #[tokio::test]
    async fn bridge_executes_mcp_tool_calls() {
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
                &sessions,
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
                .and_then(|message| message.tool_result.clone()),
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
                &sessions,
            )
            .await
            .unwrap();

        assert_eq!(
            result.response.content,
            "I could not get search results from the tool."
        );
        assert!(result.response.tool_calls.is_empty());
        assert!(result.messages.iter().any(|message| message.role == "tool"));
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
                &sessions,
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
            Err(AiError::Api(
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
            CONFIG.get_or_init(|| AiConfig {
                api_key: "test".to_string(),
                base_url: "https://example.com".to_string(),
                model: "mock".to_string(),
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
                .filter(|message| message.role == "tool")
                .count(),
            1
        );

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1]
                .messages
                .last()
                .and_then(|message| message.tool_result.clone()),
            Some(json!({ "sum": 5 }))
        );

        sessions.close().await;
    }

    #[allow(dead_code)]
    fn _assert_send(_: McpManagedChatResponse) {}
}
