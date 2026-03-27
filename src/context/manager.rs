use super::{ModelContextLimits, resolve_model_context_limits, split_text_into_windows};
use crate::ai::{AiClient, AiError, ChatRequest, ChatResponse, Message, ToolDefinition, debug_log};
use futures_util::{
    StreamExt,
    stream::{self, BoxStream},
};
use serde_json::Value;

const SUMMARY_OPEN: &str = "<conect_llm-context-summary>";
const SUMMARY_CLOSE: &str = "</conect_llm-context-summary>";

#[derive(Debug, Clone)]
pub struct ContextManagerConfig {
    pub reserve_output_tokens: u32,
    pub target_input_ratio: f32,
    pub preserve_recent_messages: usize,
    pub min_recent_messages: usize,
    pub summary_chunk_chars: usize,
    pub summary_max_output_tokens: u32,
    pub max_compaction_rounds: usize,
    pub max_message_excerpt_chars: usize,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self {
            reserve_output_tokens: 8_192,
            target_input_ratio: 0.9,
            preserve_recent_messages: 8,
            min_recent_messages: 2,
            summary_chunk_chars: 24_000,
            summary_max_output_tokens: 2_048,
            max_compaction_rounds: 3,
            max_message_excerpt_chars: 8_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCompaction {
    pub rounds: usize,
    pub summarized_messages: usize,
    pub kept_messages: usize,
    pub estimated_tokens_before: usize,
    pub estimated_tokens_after: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedChatRequest {
    pub request: ChatRequest,
    pub compaction: Option<ContextCompaction>,
}

#[derive(Debug, Clone)]
pub struct ManagedChatResponse {
    pub response: ChatResponse,
    pub compaction: Option<ContextCompaction>,
}

#[derive(Debug, Clone, Default)]
pub struct ContextManager {
    config: ContextManagerConfig,
}

impl ContextManager {
    pub fn new(config: ContextManagerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ContextManagerConfig {
        &self.config
    }

    pub fn model_limits(&self, client: &dyn AiClient, request: &ChatRequest) -> ModelContextLimits {
        resolve_model_context_limits(&client.config().base_url, &request.model)
    }

    pub fn estimate_request_tokens(&self, request: &ChatRequest) -> usize {
        let mut chars = request.model.chars().count();
        chars += request
            .system
            .as_deref()
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or(0);

        for tool in &request.tools {
            chars += tool.name.chars().count();
            chars += tool
                .description
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0);
            chars += serde_json::to_string(&tool.input_schema)
                .map(|text| text.chars().count())
                .unwrap_or_default();
        }

        for message in &request.messages {
            chars += self.estimate_message_chars(message);
        }

        (chars / 4).max(1)
    }

    pub async fn prepare_request(
        &self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<PreparedChatRequest, AiError> {
        self.prepare_request_inner(client, request, false).await
    }

    pub async fn prepare_stream_request(
        &self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<PreparedChatRequest, AiError> {
        self.prepare_request(client, request).await
    }

    pub async fn chat(
        &self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> Result<ManagedChatResponse, AiError> {
        let original_request = request.clone();
        let mut prepared = self.prepare_request(client, request).await?;

        match client.chat(prepared.request.clone()).await {
            Ok(response) => {
                return Ok(ManagedChatResponse {
                    response,
                    compaction: prepared.compaction,
                });
            }
            Err(error) if self.is_context_overflow(&error) && prepared.compaction.is_none() => {
                debug_log(
                    "context manager",
                    "context overflow detected, retrying after forced compaction",
                );
            }
            Err(error) => return Err(error),
        }

        prepared = self
            .prepare_request_inner(client, original_request, true)
            .await?;
        let response = client.chat(prepared.request.clone()).await?;
        Ok(ManagedChatResponse {
            response,
            compaction: prepared.compaction,
        })
    }

    pub fn chat_stream(
        &self,
        client: &dyn AiClient,
        request: ChatRequest,
    ) -> BoxStream<'static, Result<crate::ai::StreamChunk, AiError>> {
        let _ = client;
        let _ = request;
        stream::once(async {
            Err(AiError::Api(
                "ContextManager::chat_stream is not supported directly. Call prepare_stream_request() and then stream with the prepared request.".to_string(),
            ))
        })
        .boxed()
    }

    async fn prepare_request_inner(
        &self,
        client: &dyn AiClient,
        mut request: ChatRequest,
        force: bool,
    ) -> Result<PreparedChatRequest, AiError> {
        let limits = self.model_limits(client, &request);
        let estimated_before = self.estimate_request_tokens(&request);

        if !force && !self.should_compact(&request, limits, estimated_before) {
            return Ok(PreparedChatRequest {
                request,
                compaction: None,
            });
        }

        let mut rounds = 0usize;
        let mut summarized_messages = 0usize;
        let mut compaction_applied = false;

        while rounds < self.config.max_compaction_rounds {
            let keep = self.keep_message_count(rounds);
            let Some(compacted) = self.compact_once(client, &request, keep).await? else {
                break;
            };

            summarized_messages = summarized_messages.max(compacted.1);
            request = compacted.0;
            rounds += 1;
            compaction_applied = true;

            let estimated = self.estimate_request_tokens(&request);
            if !force && !self.should_compact(&request, limits, estimated) {
                break;
            }
        }

        let estimated_after = self.estimate_request_tokens(&request);
        let compaction = compaction_applied.then_some(ContextCompaction {
            rounds,
            summarized_messages,
            kept_messages: request.messages.len(),
            estimated_tokens_before: estimated_before,
            estimated_tokens_after: estimated_after,
        });

        if let Some(info) = &compaction {
            debug_log(
                "context manager",
                &format!(
                    "compacted context: rounds={}, summarized_messages={}, estimated_tokens {} -> {}",
                    info.rounds,
                    info.summarized_messages,
                    info.estimated_tokens_before,
                    info.estimated_tokens_after
                ),
            );
        }

        Ok(PreparedChatRequest {
            request,
            compaction,
        })
    }

    fn keep_message_count(&self, round: usize) -> usize {
        let mut keep = self.config.preserve_recent_messages;
        for _ in 0..round {
            keep = keep.saturating_sub(2);
        }
        keep.max(self.config.min_recent_messages)
    }

    fn should_compact(
        &self,
        request: &ChatRequest,
        limits: ModelContextLimits,
        estimated_tokens: usize,
    ) -> bool {
        let Some(context_window) = limits.context_window else {
            return false;
        };

        let reserve_output = request
            .max_tokens
            .or(limits.max_output_tokens)
            .unwrap_or(self.config.reserve_output_tokens)
            .min(if request.max_tokens.is_some() {
                u32::MAX
            } else {
                self.config.reserve_output_tokens
            });

        let usable = ((context_window as f32) * self.config.target_input_ratio) as usize;
        estimated_tokens.saturating_add(reserve_output as usize) >= usable
    }

    async fn compact_once(
        &self,
        client: &dyn AiClient,
        request: &ChatRequest,
        keep_recent_messages: usize,
    ) -> Result<Option<(ChatRequest, usize)>, AiError> {
        if request.messages.len() <= keep_recent_messages {
            return Ok(None);
        }

        let split_at = request.messages.len().saturating_sub(keep_recent_messages);
        let older_messages = &request.messages[..split_at];
        let recent_messages = request.messages[split_at..].to_vec();
        if older_messages.is_empty() {
            return Ok(None);
        }

        let summary = self
            .summarize_messages(client, &request.model, older_messages)
            .await?;

        let mut compacted = request.clone();
        compacted.messages = recent_messages;
        compacted.system = Some(merge_summary_into_system(
            compacted.system.as_deref(),
            &summary,
        ));

        Ok(Some((compacted, older_messages.len())))
    }

    async fn summarize_messages(
        &self,
        client: &dyn AiClient,
        model: &str,
        messages: &[Message],
    ) -> Result<String, AiError> {
        let transcript = messages
            .iter()
            .map(|message| self.render_message_for_summary(message))
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut summaries = split_text_into_windows(
            &transcript,
            super::text::TextWindowConfig {
                max_chars: self.config.summary_chunk_chars,
                overlap_chars: 0,
            },
        )
        .into_iter()
        .map(|window| window.text)
        .collect::<Vec<_>>();

        if summaries.is_empty() {
            return Ok(String::new());
        }

        while summaries.len() > 1 {
            let mut next = Vec::new();
            for chunk in summaries {
                next.push(self.summarize_chunk(client, model, &chunk).await?);
            }
            summaries = if next.len() > 1 {
                split_text_into_windows(
                    &next.join("\n\n"),
                    super::text::TextWindowConfig {
                        max_chars: self.config.summary_chunk_chars,
                        overlap_chars: 0,
                    },
                )
                .into_iter()
                .map(|window| window.text)
                .collect()
            } else {
                next
            };
        }

        self.summarize_chunk(client, model, &summaries[0]).await
    }

    async fn summarize_chunk(
        &self,
        client: &dyn AiClient,
        model: &str,
        transcript: &str,
    ) -> Result<String, AiError> {
        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![Message::user(transcript.to_string())],
            tools: Vec::<ToolDefinition>::new(),
            tool_choice: None,
            max_tokens: Some(self.config.summary_max_output_tokens),
            temperature: None,
            system: Some(
                "Summarize the earlier conversation context for continuation. Keep concrete facts, decisions, file names, tool outcomes, unresolved work, and user instructions. Be concise and structured.".to_string(),
            ),
            thinking: None,
        };

        let response = client.chat(request).await?;
        Ok(response.content.trim().to_string())
    }

    fn render_message_for_summary(&self, message: &Message) -> String {
        let mut lines = vec![format!("[{}]", message.role)];

        if !message.content.is_empty() {
            lines.push(truncate_chars(
                &message.content,
                self.config.max_message_excerpt_chars,
            ));
        }

        if let Some(thinking) = &message.thinking {
            if let Some(text) = &thinking.text {
                lines.push(format!(
                    "thinking: {}",
                    truncate_chars(text, self.config.max_message_excerpt_chars / 2)
                ));
            }
        }

        if !message.tool_calls.is_empty() {
            for tool_call in &message.tool_calls {
                lines.push(format!(
                    "tool call {} {} {}",
                    tool_call.id,
                    tool_call.name,
                    truncate_chars(
                        &serde_json::to_string(&tool_call.arguments)
                            .unwrap_or_else(|_| tool_call.arguments.to_string()),
                        self.config.max_message_excerpt_chars / 2
                    )
                ));
            }
        }

        if let (Some(tool_name), Some(tool_result)) = (&message.tool_name, &message.tool_result) {
            lines.push(format!(
                "tool result {} {}",
                tool_name,
                truncate_chars(
                    &serialize_value(tool_result),
                    self.config.max_message_excerpt_chars / 2
                )
            ));
        }

        lines.join("\n")
    }

    fn estimate_message_chars(&self, message: &Message) -> usize {
        let mut chars = message.role.chars().count() + message.content.chars().count();
        if let Some(thinking) = &message.thinking {
            chars += thinking
                .text
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0);
            chars += thinking
                .signature
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0);
            chars += thinking
                .redacted
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0);
        }
        if let Some(tool_name) = &message.tool_name {
            chars += tool_name.chars().count();
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            chars += tool_call_id.chars().count();
        }
        if let Some(tool_result) = &message.tool_result {
            chars += serialize_value(tool_result).chars().count();
        }
        for tool_call in &message.tool_calls {
            chars += tool_call.id.chars().count();
            chars += tool_call.name.chars().count();
            chars += serialize_value(&tool_call.arguments).chars().count();
        }
        chars + 32
    }

    fn is_context_overflow(&self, error: &AiError) -> bool {
        let message = error.to_string().to_ascii_lowercase();
        [
            "prompt is too long",
            "input is too long for requested model",
            "exceeds the context window",
            "input token count",
            "maximum context length is",
            "exceeds the available context size",
            "greater than the context length",
            "context window exceeds limit",
            "exceeded model token limit",
            "context length exceeded",
            "context_length_exceeded",
            "input exceeds context window",
        ]
        .iter()
        .any(|pattern| message.contains(pattern))
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let head: String = text.chars().take(max_chars).collect();
    format!("{}...", head)
}

fn serialize_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn merge_summary_into_system(existing: Option<&str>, summary: &str) -> String {
    let stripped = strip_existing_summary(existing.unwrap_or_default());
    if stripped.trim().is_empty() {
        return format!("{}\n{}\n{}", SUMMARY_OPEN, summary.trim(), SUMMARY_CLOSE);
    }

    format!(
        "{}\n\n{}\n{}\n{}",
        stripped.trim_end(),
        SUMMARY_OPEN,
        summary.trim(),
        SUMMARY_CLOSE
    )
}

fn strip_existing_summary(system: &str) -> String {
    if let Some(start) = system.find(SUMMARY_OPEN) {
        if let Some(end) = system.find(SUMMARY_CLOSE) {
            let end = end + SUMMARY_CLOSE.len();
            let mut stripped = String::new();
            stripped.push_str(&system[..start]);
            stripped.push_str(&system[end..]);
            return stripped.trim().to_string();
        }
    }

    system.to_string()
}

#[cfg(test)]
mod tests {
    use super::{ContextManager, ContextManagerConfig, merge_summary_into_system};
    use crate::ai::{
        AiClient, AiConfig, AiError, ChatRequest, ChatResponse, Message, StreamChunk, Usage,
    };
    use futures_util::{
        StreamExt,
        stream::{self, BoxStream},
    };
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct MockClient {
        config: AiConfig,
        calls: Arc<Mutex<Vec<ChatRequest>>>,
        fail_large_once: bool,
    }

    #[async_trait::async_trait]
    impl AiClient for MockClient {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, AiError> {
            self.calls.lock().unwrap().push(request.clone());
            if self.fail_large_once
                && self.calls.lock().unwrap().len() == 1
                && request.messages.len() > 4
            {
                return Err(AiError::Api("prompt is too long".to_string()));
            }
            if request
                .system
                .as_deref()
                .unwrap_or_default()
                .contains("Summarize the earlier conversation context")
            {
                return Ok(ChatResponse {
                    id: "summary".to_string(),
                    content: "Summarized earlier context.".to_string(),
                    model: request.model,
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    thinking: None,
                    tool_calls: Vec::new(),
                    debug: None,
                });
            }
            Ok(ChatResponse {
                id: "ok".to_string(),
                content: "done".to_string(),
                model: request.model,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 10,
                },
                thinking: None,
                tool_calls: Vec::new(),
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
            &self.config
        }

        async fn list_models(&self) -> Result<Vec<String>, AiError> {
            Ok(vec![self.config.model.clone()])
        }
    }

    #[tokio::test]
    async fn prepare_request_compacts_large_history() {
        let manager = ContextManager::new(ContextManagerConfig {
            reserve_output_tokens: 2048,
            preserve_recent_messages: 4,
            min_recent_messages: 2,
            ..Default::default()
        });
        let client = MockClient {
            config: AiConfig {
                api_key: "test".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            calls: Arc::new(Mutex::new(Vec::new())),
            fail_large_once: false,
        };

        let messages = (0..12)
            .map(|index| Message::user("x".repeat(80_000 + index)))
            .collect();
        let request = ChatRequest::new("claude-sonnet-4-20250514", messages);

        let prepared = manager.prepare_request(&client, request).await.unwrap();
        assert!(prepared.compaction.is_some());
        assert!(prepared.request.messages.len() <= 4);
        assert!(
            prepared
                .request
                .system
                .as_deref()
                .unwrap_or_default()
                .contains("Summarized earlier context.")
        );
    }

    #[tokio::test]
    async fn chat_retries_after_context_overflow() {
        let manager = ContextManager::default();
        let client = MockClient {
            config: AiConfig {
                api_key: "test".to_string(),
                base_url: "https://unknown.example.com".to_string(),
                model: "unknown".to_string(),
            },
            calls: Arc::new(Mutex::new(Vec::new())),
            fail_large_once: true,
        };

        let messages = (0..10)
            .map(|index| Message::user("y".repeat(5000 + index)))
            .collect();
        let request = ChatRequest::new("unknown", messages);

        let response = manager.chat(&client, request).await.unwrap();
        assert_eq!(response.response.content, "done");
        assert!(response.compaction.is_some());
        assert!(client.calls.lock().unwrap().len() >= 3);
    }

    #[test]
    fn merges_summary_sections_idempotently() {
        let merged = merge_summary_into_system(
            Some(
                "instructions\n\n<conect_llm-context-summary>\nold\n</conect_llm-context-summary>",
            ),
            "new",
        );
        assert!(merged.contains("instructions"));
        assert!(merged.contains("new"));
        assert!(!merged.contains("old"));
    }
}
