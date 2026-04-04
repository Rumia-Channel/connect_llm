use crate::{
    ai::{AiClient, AiError, ChatRequest, ChatResponse, Message, ToolCall, ToolDefinition},
    context::{ContextCompaction, ContextManager, ManagedChatResponse},
};
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
    path::PathBuf,
};
use tokio::process::Command;

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

    pub fn is_empty(&self) -> bool {
        self.mcp_servers.values().all(|server| !server.enabled) || self.mcp_servers.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
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
    pub description: Option<String>,
    pub transport: Option<String>,
    #[serde(default, alias = "authorization")]
    pub auth_header: Option<String>,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: None,
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            url: None,
            headers: BTreeMap::new(),
            description: None,
            transport: None,
            auth_header: None,
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolExecution {
    pub tool_call_id: String,
    pub tool_name: String,
    pub server_label: String,
    pub remote_tool_name: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub struct McpBridge {
    config: McpConfig,
    tool_loop: McpToolLoopConfig,
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

        for round in 0..=self.tool_loop.max_round_trips {
            let managed = send_chat(context_manager, client, request.clone()).await?;
            let compaction = managed.compaction.clone();

            let response = managed.response;
            let assistant_message = assistant_message_from_response(&response);
            let tool_calls = response.tool_calls.clone();
            request.messages.push(assistant_message);

            if tool_calls.is_empty() {
                return Ok(McpManagedChatResponse {
                    response,
                    compaction,
                    messages: request.messages,
                    tool_executions,
                });
            }

            if round == self.tool_loop.max_round_trips {
                return Err(AiError::Api(format!(
                    "MCP tool loop exceeded {} round trips",
                    self.tool_loop.max_round_trips
                )));
            }

            let (tool_results, executed) = sessions.execute_tool_calls(&tool_calls).await?;
            request.messages.extend(tool_results);
            tool_executions.extend(executed);
        }

        Err(AiError::Api(
            "MCP tool loop terminated unexpectedly".to_string(),
        ))
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
}

impl McpSessionSet {
    async fn connect(config: &McpConfig) -> Result<Self, AiError> {
        if config.is_empty() {
            return Ok(Self {
                servers: Vec::new(),
                exported_tools: Vec::new(),
                tool_index: HashMap::new(),
            });
        }

        let mut pending = Vec::new();
        for (label, server) in &config.mcp_servers {
            if !server.enabled {
                continue;
            }
            pending.push(PendingMcpServer::connect(label, server).await?);
        }

        Ok(Self::from_pending(pending))
    }

    fn from_pending(pending: Vec<PendingMcpServer>) -> Self {
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
                service: pending_server.service,
                peer: pending_server.peer,
                aliases,
            });
        }

        Self {
            servers,
            exported_tools,
            tool_index,
        }
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.exported_tools.clone()
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
            let is_error = result.is_error.unwrap_or(false);
            let mut message = Message::tool_result(
                tool_call.id.clone(),
                tool_call.name.clone(),
                call_tool_result_to_value(&result),
            );
            if is_error {
                message.tool_error = Some(true);
            }
            messages.push(message);
            executions.push(McpToolExecution {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                server_label: resolved.server_label.clone(),
                remote_tool_name,
                is_error,
            });
        }

        Ok((messages, executions))
    }
}

struct McpServerSession {
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

async fn connect_peer(config: &McpServerConfig) -> Result<RunningService<RoleClient, ()>, AiError> {
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
            let transport =
                TokioChildProcess::new(process).map_err(|error| AiError::Api(error.to_string()))?;
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
            if !config.headers.is_empty() {
                transport_config =
                    transport_config.custom_headers(build_custom_headers(&config.headers)?);
            }
            let transport = StreamableHttpClientTransport::from_config(transport_config);
            ().serve(transport).await.map_err(mcp_service_error)
        }
    }
}

fn detect_transport(config: &McpServerConfig) -> Result<McpTransport, AiError> {
    if let Some(transport) = config.transport.as_deref() {
        return match transport {
            "stdio" => Ok(McpTransport::Stdio),
            "http" | "streamable_http" | "sse" => Ok(McpTransport::StreamableHttp),
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

fn value_as_json_object(value: &Value) -> Result<Map<String, Value>, AiError> {
    match value {
        Value::Object(object) => Ok(object.clone()),
        other => Err(AiError::Parse(format!(
            "MCP tool arguments must be a JSON object, got {}",
            other
        ))),
    }
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
        Usage,
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
    use std::sync::{Arc, Mutex};

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

    #[tokio::test]
    async fn bridge_executes_mcp_tool_calls() {
        let (server_transport, client_transport) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            let server = TestToolServer.serve(server_transport).await.unwrap();
            server.waiting().await.unwrap();
        });

        let service = ().serve(client_transport).await.unwrap();
        let sessions = super::McpSessionSet::from_pending(vec![
            PendingMcpServer::from_service(
                "calc".to_string(),
                Some("Calculator".to_string()),
                service,
            )
            .await
            .unwrap(),
        ]);
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

    #[allow(dead_code)]
    fn _assert_send(_: McpManagedChatResponse) {}
}
