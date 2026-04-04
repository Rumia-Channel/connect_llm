use super::{ModelContextLimits, resolve_model_context_limits, split_text_into_windows};
use crate::ai::{AiClient, AiError, ChatRequest, ChatResponse, Message, ToolDefinition, debug_log};
use futures_util::{
    StreamExt,
    stream::{self, BoxStream},
};
use serde_json::Value;
use std::{collections::HashSet, time::Duration};

const SUMMARY_OPEN: &str = "<connect_llm-context-summary>";
const SUMMARY_CLOSE: &str = "</connect_llm-context-summary>";
const MIN_RECENT_TOKEN_BUDGET: usize = 2_048;
const SUMMARY_RETRY_ATTEMPTS: usize = 3;
const SUMMARY_RETRY_MARKER: &str = "[earlier summary chunk truncated for retry]";

#[derive(Debug, Clone)]
pub struct ContextManagerConfig {
    pub reserve_output_tokens: u32,
    pub target_input_ratio: f32,
    pub preserve_recent_messages: usize,
    pub min_recent_messages: usize,
    pub stale_message_age: Duration,
    pub session_gap: Duration,
    pub summary_chunk_chars: usize,
    pub summary_max_output_tokens: u32,
    pub max_compaction_rounds: usize,
    pub max_message_excerpt_chars: usize,
    pub stale_text_excerpt_chars: usize,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self {
            reserve_output_tokens: 8_192,
            target_input_ratio: 0.9,
            preserve_recent_messages: 8,
            min_recent_messages: 2,
            stale_message_age: Duration::from_secs(15 * 60),
            session_gap: Duration::from_secs(30 * 60),
            summary_chunk_chars: 24_000,
            summary_max_output_tokens: 2_048,
            max_compaction_rounds: 3,
            max_message_excerpt_chars: 8_000,
            stale_text_excerpt_chars: 1_200,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCompaction {
    pub rounds: usize,
    pub summarized_messages: usize,
    pub microcompaction_passes: usize,
    pub microcompacted_messages: usize,
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
            Err(error)
                if self.is_context_overflow(&error)
                    && prepared
                        .compaction
                        .as_ref()
                        .is_none_or(|compaction| compaction.rounds == 0) =>
            {
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
        let mut microcompaction_passes = 0usize;
        let mut microcompacted_messages = 0usize;
        let mut compaction_applied = false;

        while rounds < self.config.max_compaction_rounds {
            if rounds == 0 {
                let recent_start =
                    self.compaction_recent_messages_start(&request, limits, rounds, force);
                let (microcompacted, changed_messages, passes) =
                    self.microcompact_request(&request, recent_start, limits, force);
                if changed_messages > 0 {
                    let compacted_estimate = self.estimate_request_tokens(&microcompacted);
                    debug_log(
                        "context manager",
                        &format!(
                            "microcompact applied {} passes across {} messages; estimated tokens {} -> {}",
                            passes, changed_messages, estimated_before, compacted_estimate
                        ),
                    );
                    request = microcompacted;
                    microcompaction_passes = passes;
                    microcompacted_messages = changed_messages;
                    compaction_applied = true;
                    if !force && !self.should_compact(&request, limits, compacted_estimate) {
                        break;
                    }
                }
            }

            let recent_start =
                self.compaction_recent_messages_start(&request, limits, rounds, force);
            let Some(compacted) = self.compact_once(client, &request, recent_start).await? else {
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
            microcompaction_passes,
            microcompacted_messages,
            kept_messages: request.messages.len(),
            estimated_tokens_before: estimated_before,
            estimated_tokens_after: estimated_after,
        });

        if let Some(info) = &compaction {
            debug_log(
                "context manager",
                &format!(
                    "compacted context: rounds={}, summarized_messages={}, microcompacted_messages={}, estimated_tokens {} -> {}",
                    info.rounds,
                    info.summarized_messages,
                    info.microcompacted_messages,
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

    fn resolved_output_reserve(&self, request: &ChatRequest, limits: ModelContextLimits) -> usize {
        request
            .max_tokens
            .or(limits.max_output_tokens)
            .unwrap_or(self.config.reserve_output_tokens)
            .min(if request.max_tokens.is_some() {
                u32::MAX
            } else {
                self.config.reserve_output_tokens
            }) as usize
    }

    fn recent_messages_start(
        &self,
        request: &ChatRequest,
        limits: ModelContextLimits,
        round: usize,
    ) -> usize {
        let fallback_keep = self.keep_message_count(round);
        let Some(context_window) = limits.context_window else {
            return request.messages.len().saturating_sub(fallback_keep);
        };

        let usable = ((context_window as f32) * self.config.target_input_ratio) as usize;
        let reserve_output = self.resolved_output_reserve(request, limits);
        let reduction_step = (usable / 20).max(1_024);
        let keep_budget = (usable / 4)
            .max(reserve_output.saturating_mul(2))
            .saturating_sub(reduction_step.saturating_mul(round))
            .max(MIN_RECENT_TOKEN_BUDGET);

        self.find_recent_messages_start(&request.messages, keep_budget, fallback_keep)
    }

    fn compaction_recent_messages_start(
        &self,
        request: &ChatRequest,
        limits: ModelContextLimits,
        round: usize,
        force: bool,
    ) -> usize {
        let mut start = self.recent_messages_start(request, limits, round);
        if let Some(boundary) = self.latest_session_boundary(&request.messages) {
            start =
                self.adjust_recent_start_to_preserve_links(&request.messages, start.max(boundary));
        }
        if !force || start > 0 {
            return start;
        }

        request
            .messages
            .len()
            .saturating_sub(self.keep_message_count(round))
    }

    fn latest_session_boundary(&self, messages: &[Message]) -> Option<usize> {
        let session_gap_ms = duration_to_millis(self.config.session_gap)?;
        if session_gap_ms == 0 {
            return None;
        }

        let mut boundary = None;
        for index in 1..messages.len() {
            let previous = messages[index - 1].created_at_ms?;
            let current = messages[index].created_at_ms?;
            if current.saturating_sub(previous) >= session_gap_ms {
                boundary = Some(index);
            }
        }

        boundary
    }

    fn latest_message_timestamp_ms(&self, messages: &[Message]) -> Option<u64> {
        messages
            .iter()
            .filter_map(|message| message.created_at_ms)
            .max()
    }

    fn is_stale_for_microcompact(
        &self,
        message: &Message,
        latest_timestamp_ms: Option<u64>,
    ) -> bool {
        let Some(latest_timestamp_ms) = latest_timestamp_ms else {
            return false;
        };
        let Some(created_at_ms) = message.created_at_ms else {
            return false;
        };
        let Some(stale_age_ms) = duration_to_millis(self.config.stale_message_age) else {
            return false;
        };
        latest_timestamp_ms.saturating_sub(created_at_ms) >= stale_age_ms
    }

    fn find_recent_messages_start(
        &self,
        messages: &[Message],
        token_budget: usize,
        min_messages: usize,
    ) -> usize {
        if messages.len() <= min_messages {
            return 0;
        }

        let mut start = messages.len();
        let mut kept_messages = 0usize;
        let mut kept_tokens = 0usize;

        while start > 0 {
            let candidate = start - 1;
            let candidate_tokens = self.estimate_message_tokens(&messages[candidate]);
            if kept_messages >= min_messages
                && kept_tokens.saturating_add(candidate_tokens) > token_budget
            {
                break;
            }

            start = candidate;
            kept_messages += 1;
            kept_tokens = kept_tokens.saturating_add(candidate_tokens);
        }

        self.adjust_recent_start_to_preserve_links(messages, start)
    }

    fn adjust_recent_start_to_preserve_links(&self, messages: &[Message], start: usize) -> usize {
        if start == 0 || start >= messages.len() {
            return start;
        }

        let mut adjusted = start;
        let mut available_tool_calls = collect_tool_call_ids(&messages[adjusted..]);
        let mut required_tool_calls = collect_required_tool_call_ids(&messages[adjusted..]);
        required_tool_calls.retain(|tool_call_id| !available_tool_calls.contains(tool_call_id));

        while !required_tool_calls.is_empty() && adjusted > 0 {
            adjusted -= 1;
            for tool_call in &messages[adjusted].tool_calls {
                available_tool_calls.insert(tool_call.id.clone());
                required_tool_calls.remove(&tool_call.id);
            }
        }

        adjusted
    }

    fn microcompact_request(
        &self,
        request: &ChatRequest,
        recent_start: usize,
        limits: ModelContextLimits,
        force: bool,
    ) -> (ChatRequest, usize, usize) {
        if recent_start == 0 {
            return (request.clone(), 0, 0);
        }

        let mut compacted_request = request.clone();
        let mut changed_messages = 0usize;
        let mut passes = 0usize;
        let latest_timestamp_ms = self.latest_message_timestamp_ms(&compacted_request.messages);

        while let Some(batch) = self.next_microcompact_batch(
            &compacted_request.messages,
            recent_start,
            latest_timestamp_ms,
        ) {
            let mut changed_this_pass = 0usize;
            for index in batch {
                let (compacted, changed) = self
                    .microcompact_message(&compacted_request.messages[index], latest_timestamp_ms);
                if changed {
                    compacted_request.messages[index] = compacted;
                    changed_messages += 1;
                    changed_this_pass += 1;
                }
            }

            if changed_this_pass == 0 {
                break;
            }

            passes += 1;
            let estimated = self.estimate_request_tokens(&compacted_request);
            if !force && !self.should_compact(&compacted_request, limits, estimated) {
                break;
            }
        }

        if changed_messages == 0 {
            return (request.clone(), 0, 0);
        }

        (compacted_request, changed_messages, passes)
    }

    fn next_microcompact_batch(
        &self,
        messages: &[Message],
        recent_start: usize,
        latest_timestamp_ms: Option<u64>,
    ) -> Option<Vec<usize>> {
        for index in 0..recent_start.min(messages.len()) {
            let message = &messages[index];
            if message.role == "tool"
                && self.message_needs_microcompact(message, latest_timestamp_ms)
            {
                let mut batch = Vec::with_capacity(2);
                if let Some(tool_call_id) = message.tool_call_id.as_deref() {
                    if let Some(tool_call_index) =
                        find_tool_call_message_index(messages, tool_call_id, index)
                    {
                        batch.push(tool_call_index);
                    }
                }
                batch.push(index);
                batch.sort_unstable();
                batch.dedup();
                return Some(batch);
            }

            if self.message_needs_microcompact(message, latest_timestamp_ms) {
                return Some(vec![index]);
            }
        }

        None
    }

    fn message_needs_microcompact(
        &self,
        message: &Message,
        latest_timestamp_ms: Option<u64>,
    ) -> bool {
        if message.thinking.is_some() {
            return true;
        }

        if self.is_stale_for_microcompact(message, latest_timestamp_ms)
            && compact_text_for_request(&message.content, self.config.stale_text_excerpt_chars)
                != message.content
        {
            return true;
        }

        if let Some(tool_result) = &message.tool_result {
            if compact_value_for_request(tool_result, self.config.max_message_excerpt_chars / 2)
                != *tool_result
            {
                return true;
            }
        }

        message.tool_calls.iter().any(|tool_call| {
            compact_value_for_request(
                &tool_call.arguments,
                self.config.max_message_excerpt_chars / 2,
            ) != tool_call.arguments
        })
    }

    fn microcompact_message(
        &self,
        message: &Message,
        latest_timestamp_ms: Option<u64>,
    ) -> (Message, bool) {
        let mut compacted = message.clone();
        let mut changed = false;

        if self.is_stale_for_microcompact(message, latest_timestamp_ms) {
            let compacted_content =
                compact_text_for_request(&compacted.content, self.config.stale_text_excerpt_chars);
            if compacted_content != compacted.content {
                compacted.content = compacted_content;
                changed = true;
            }
        }

        if compacted.thinking.take().is_some() {
            changed = true;
        }

        if let Some(tool_result) = &compacted.tool_result {
            let compacted_value =
                compact_value_for_request(tool_result, self.config.max_message_excerpt_chars / 2);
            if compacted_value != *tool_result {
                compacted.tool_result = Some(compacted_value);
                changed = true;
            }
        }

        for tool_call in &mut compacted.tool_calls {
            let compacted_arguments = compact_value_for_request(
                &tool_call.arguments,
                self.config.max_message_excerpt_chars / 2,
            );
            if compacted_arguments != tool_call.arguments {
                tool_call.arguments = compacted_arguments;
                changed = true;
            }
        }

        (compacted, changed)
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

        let reserve_output = self.resolved_output_reserve(request, limits);

        let usable = ((context_window as f32) * self.config.target_input_ratio) as usize;
        estimated_tokens.saturating_add(reserve_output) >= usable
    }

    async fn compact_once(
        &self,
        client: &dyn AiClient,
        request: &ChatRequest,
        recent_start: usize,
    ) -> Result<Option<(ChatRequest, usize)>, AiError> {
        if recent_start == 0 || request.messages.len() <= recent_start {
            return Ok(None);
        }

        let older_messages = &request.messages[..recent_start];
        let recent_messages = request.messages[recent_start..].to_vec();
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
        let mut current = transcript.to_string();
        for attempt in 0..=SUMMARY_RETRY_ATTEMPTS {
            let request = ChatRequest {
                model: model.to_string(),
                messages: vec![Message::user(current.clone())],
                tools: Vec::<ToolDefinition>::new(),
                tool_choice: None,
                max_tokens: Some(self.config.summary_max_output_tokens),
                temperature: None,
                system: Some(
                    "Summarize the earlier conversation context for continuation. Keep concrete facts, decisions, file names, tool outcomes, unresolved work, and user instructions. Be concise and structured.".to_string(),
                ),
                thinking: None,
            };

            match client.chat(request).await {
                Ok(response) => return Ok(response.content.trim().to_string()),
                Err(error)
                    if attempt < SUMMARY_RETRY_ATTEMPTS && self.is_context_overflow(&error) =>
                {
                    let Some(next) = truncate_summary_transcript_for_retry(&current) else {
                        return Err(error);
                    };
                    debug_log(
                        "context manager",
                        &format!(
                            "summary chunk overflowed on attempt {}; retrying with shorter transcript",
                            attempt + 1
                        ),
                    );
                    current = next;
                }
                Err(error) => return Err(error),
            }
        }

        unreachable!("summary retry loop must return")
    }

    fn render_message_for_summary(&self, message: &Message) -> String {
        let mut lines = vec![format!("[{}]", message.role)];

        if !message.content.is_empty() {
            lines.push(truncate_chars(
                &message.content,
                self.config.max_message_excerpt_chars,
            ));
        }

        if !message.tool_calls.is_empty() {
            for tool_call in &message.tool_calls {
                lines.push(format!(
                    "tool call {} {} {}",
                    tool_call.id,
                    tool_call.name,
                    preview_value_for_summary(
                        &tool_call.arguments,
                        self.config.max_message_excerpt_chars / 2
                    )
                ));
            }
        }

        if let (Some(tool_name), Some(tool_result)) = (&message.tool_name, &message.tool_result) {
            lines.push(format!(
                "tool result {} {}",
                tool_name,
                preview_value_for_summary(tool_result, self.config.max_message_excerpt_chars / 2)
            ));
        }

        lines.join("\n")
    }

    fn estimate_message_tokens(&self, message: &Message) -> usize {
        (self.estimate_message_chars(message) / 4).max(1)
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

fn collect_tool_call_ids(messages: &[Message]) -> HashSet<String> {
    messages
        .iter()
        .flat_map(|message| {
            message
                .tool_calls
                .iter()
                .map(|tool_call| tool_call.id.clone())
        })
        .collect()
}

fn collect_required_tool_call_ids(messages: &[Message]) -> HashSet<String> {
    messages
        .iter()
        .filter_map(|message| {
            (message.role == "tool")
                .then(|| message.tool_call_id.clone())
                .flatten()
        })
        .collect()
}

fn find_tool_call_message_index(
    messages: &[Message],
    tool_call_id: &str,
    before_index: usize,
) -> Option<usize> {
    messages[..before_index]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            message
                .tool_calls
                .iter()
                .any(|tool_call| tool_call.id == tool_call_id)
                .then_some(index)
        })
}

fn truncate_summary_transcript_for_retry(transcript: &str) -> Option<String> {
    let len = transcript.chars().count();
    if len <= 1_024 {
        return None;
    }

    let drop_chars = (len / 5).max(512).min(len.saturating_sub(512));
    let tail: String = transcript.chars().skip(drop_chars).collect();
    Some(format!("{}\n\n{}", SUMMARY_RETRY_MARKER, tail))
}

fn duration_to_millis(duration: Duration) -> Option<u64> {
    u64::try_from(duration.as_millis()).ok()
}

fn compact_text_for_request(text: &str, max_chars: usize) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars || max_chars < 64 {
        return text.to_string();
    }

    let head_chars = (max_chars * 2 / 3).max(48).min(total_chars);
    let tail_chars = max_chars.saturating_sub(head_chars).max(24);
    let head: String = text.chars().take(head_chars).collect();
    let tail: String = text
        .chars()
        .rev()
        .take(tail_chars.min(total_chars.saturating_sub(head_chars)))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    if tail.is_empty() {
        format!(
            "[connect_llm compacted stale message excerpt ({} chars total)]\n{}",
            total_chars, head
        )
    } else {
        format!(
            "[connect_llm compacted stale message excerpt ({} chars total)]\n{}\n...\n{}",
            total_chars, head, tail
        )
    }
}

fn compact_value_for_request(value: &Value, max_chars: usize) -> Value {
    if is_value_already_compacted(value) {
        return value.clone();
    }

    let serialized = serialize_value(value);
    if serialized.chars().count() <= max_chars && !value_contains_large_binary(value) {
        return value.clone();
    }

    Value::String(format!(
        "[connect_llm compacted {}]",
        summarize_value_shape(value)
    ))
}

fn preview_value_for_summary(value: &Value, max_chars: usize) -> String {
    if is_value_already_compacted(value) {
        return value.as_str().unwrap_or_default().to_string();
    }

    if value_contains_large_binary(value) {
        return summarize_value_shape(value);
    }

    truncate_chars(&serialize_value(value), max_chars)
}

fn is_value_already_compacted(value: &Value) -> bool {
    matches!(
        value,
        Value::String(text) if text.starts_with("[connect_llm compacted ")
    )
}

fn value_contains_large_binary(value: &Value) -> bool {
    match value {
        Value::String(text) => looks_like_large_binary(text),
        Value::Array(items) => items.iter().any(value_contains_large_binary),
        Value::Object(map) => {
            map.contains_key("data_base64")
                || map.contains_key("mime_type")
                || map.contains_key("inline_data")
                || map.values().any(value_contains_large_binary)
        }
        _ => false,
    }
}

fn looks_like_large_binary(text: &str) -> bool {
    text.len() >= 512
        && text
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '-' | '_'))
}

fn summarize_value_shape(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "boolean value omitted".to_string(),
        Value::Number(_) => "numeric value omitted".to_string(),
        Value::String(text) if looks_like_large_binary(text) => {
            "binary payload omitted".to_string()
        }
        Value::String(text) => format!("string omitted ({} chars)", text.chars().count()),
        Value::Array(items) => format!("array omitted ({} items)", items.len()),
        Value::Object(map) if map.contains_key("data_base64") || map.contains_key("mime_type") => {
            "image payload omitted".to_string()
        }
        Value::Object(map) => {
            let keys = map.keys().take(5).cloned().collect::<Vec<_>>();
            if keys.is_empty() {
                "empty object omitted".to_string()
            } else {
                format!("object omitted (keys: {})", keys.join(", "))
            }
        }
    }
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
        AiClient, AiConfig, AiError, ChatRequest, ChatResponse, Message, StreamChunk, ToolCall,
        Usage,
    };
    use futures_util::{
        StreamExt,
        stream::{self, BoxStream},
    };
    use serde_json::{Value, json};
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    #[derive(Clone)]
    struct MockClient {
        config: AiConfig,
        calls: Arc<Mutex<Vec<ChatRequest>>>,
        fail_large_once: bool,
        summary_overflow_chars: Option<usize>,
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
                if let Some(max_chars) = self.summary_overflow_chars {
                    let summary_input = request
                        .messages
                        .first()
                        .map(|message| message.content.chars().count())
                        .unwrap_or_default();
                    if summary_input > max_chars {
                        return Err(AiError::Api("prompt is too long".to_string()));
                    }
                }
                return Ok(ChatResponse {
                    id: "summary".to_string(),
                    content: "Summarized earlier context.".to_string(),
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
                id: "ok".to_string(),
                content: "done".to_string(),
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
            summary_overflow_chars: None,
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
            summary_overflow_chars: None,
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
                "instructions\n\n<connect_llm-context-summary>\nold\n</connect_llm-context-summary>",
            ),
            "new",
        );
        assert!(merged.contains("instructions"));
        assert!(merged.contains("new"));
        assert!(!merged.contains("old"));
    }

    #[tokio::test]
    async fn prepare_request_microcompact_can_avoid_summary() {
        let manager = ContextManager::new(ContextManagerConfig {
            reserve_output_tokens: 2048,
            preserve_recent_messages: 2,
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
            summary_overflow_chars: None,
        };

        let mut messages = Vec::new();
        messages.push(Message::assistant_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "src/main.rs" }),
        }]));
        messages.push(Message::tool_result(
            "call_1",
            "read_file",
            Value::String("x".repeat(900_000)),
        ));
        messages.push(Message::assistant("recent assistant context"));
        messages.push(Message::user("recent user request"));

        let prepared = manager
            .prepare_request(
                &client,
                ChatRequest::new("claude-sonnet-4-20250514", messages),
            )
            .await
            .unwrap();
        let compaction = prepared.compaction.as_ref().expect("microcompact expected");
        assert_eq!(compaction.rounds, 0);
        assert_eq!(compaction.summarized_messages, 0);
        assert!(compaction.microcompacted_messages >= 1);
        assert!(compaction.microcompaction_passes >= 1);
        assert_eq!(prepared.request.messages.len(), 4);
        let tool_result = prepared.request.messages[1]
            .tool_result
            .as_ref()
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(tool_result.starts_with("[connect_llm compacted "));
        assert!(prepared.request.system.is_none());
    }

    #[tokio::test]
    async fn prepare_request_microcompacts_stale_long_text_without_summary() {
        let manager = ContextManager::new(ContextManagerConfig {
            reserve_output_tokens: 2048,
            preserve_recent_messages: 2,
            min_recent_messages: 2,
            stale_message_age: Duration::from_secs(5 * 60),
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
            summary_overflow_chars: None,
        };

        let base = 1_700_000_000_000u64;
        let messages = vec![
            Message::user("older user context ".repeat(30_000)).with_created_at_ms(base),
            Message::assistant("older assistant context ".repeat(30_000))
                .with_created_at_ms(base + 1_000),
            Message::user("recent question").with_created_at_ms(base + 12 * 60 * 1_000),
            Message::assistant("recent answer").with_created_at_ms(base + 12 * 60 * 1_000 + 500),
        ];

        let prepared = manager
            .prepare_request(
                &client,
                ChatRequest::new("claude-sonnet-4-20250514", messages),
            )
            .await
            .unwrap();

        let compaction = prepared.compaction.as_ref().expect("microcompact expected");
        assert_eq!(compaction.rounds, 0);
        assert_eq!(compaction.summarized_messages, 0);
        assert!(compaction.microcompacted_messages >= 1);
        assert!(
            prepared.request.messages[0]
                .content
                .starts_with("[connect_llm compacted stale message excerpt")
        );
        assert!(prepared.request.system.is_none());
    }

    #[test]
    fn recent_boundary_prefers_latest_session_after_long_gap() {
        let manager = ContextManager::new(ContextManagerConfig {
            preserve_recent_messages: 4,
            min_recent_messages: 2,
            session_gap: Duration::from_secs(30 * 60),
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
            summary_overflow_chars: None,
        };
        let base = 1_700_000_000_000u64;
        let request = ChatRequest::new(
            "claude-sonnet-4-20250514",
            vec![
                Message::user("session one user").with_created_at_ms(base),
                Message::assistant("session one assistant").with_created_at_ms(base + 1_000),
                Message::user("session two user").with_created_at_ms(base + 45 * 60 * 1_000),
                Message::assistant("session two assistant")
                    .with_created_at_ms(base + 45 * 60 * 1_000 + 1_000),
            ],
        );

        let limits = manager.model_limits(&client, &request);
        assert_eq!(
            manager.compaction_recent_messages_start(&request, limits, 0, false),
            2
        );
    }

    #[tokio::test]
    async fn chat_retries_with_summary_after_microcompact_only_overflow() {
        let manager = ContextManager::new(ContextManagerConfig {
            reserve_output_tokens: 2048,
            preserve_recent_messages: 2,
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
            fail_large_once: true,
            summary_overflow_chars: None,
        };

        let mut messages = vec![
            Message::assistant_tool_calls(vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "src/main.rs", "body": "x".repeat(200_000) }),
            }]),
            Message::tool_result("call_1", "read_file", Value::String("x".repeat(900_000))),
        ];
        messages.extend((0..3).map(|index| Message::user(format!("recent {}", index))));

        let response = manager
            .chat(
                &client,
                ChatRequest::new("claude-sonnet-4-20250514", messages),
            )
            .await
            .unwrap();

        assert_eq!(response.response.content, "done");
        let compaction = response
            .compaction
            .as_ref()
            .expect("summary compaction expected");
        assert!(compaction.microcompacted_messages >= 2);
        assert!(compaction.rounds >= 1);
        let calls = client.calls.lock().unwrap();
        let summary_calls = calls
            .iter()
            .filter(|request| {
                request
                    .system
                    .as_deref()
                    .unwrap_or_default()
                    .contains("Summarize the earlier conversation context")
            })
            .count();
        assert!(summary_calls >= 1);
    }

    #[tokio::test]
    async fn prepare_request_keeps_tool_call_with_recent_tool_result() {
        let manager = ContextManager::new(ContextManagerConfig {
            reserve_output_tokens: 2048,
            preserve_recent_messages: 2,
            min_recent_messages: 1,
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
            summary_overflow_chars: None,
        };

        let mut messages = (0..6)
            .map(|_| Message::user("older context ".repeat(40_000)))
            .collect::<Vec<_>>();
        messages.push(Message::assistant_tool_calls(vec![ToolCall {
            id: "call_2".to_string(),
            name: "grep".to_string(),
            arguments: json!({ "pattern": "TODO" }),
        }]));
        messages.push(Message::tool_result(
            "call_2",
            "grep",
            Value::String("hit".repeat(20_000)),
        ));

        let prepared = manager
            .prepare_request(
                &client,
                ChatRequest::new("claude-sonnet-4-20250514", messages),
            )
            .await
            .unwrap();

        assert!(prepared.compaction.is_some());
        assert_eq!(prepared.request.messages.len(), 2);
        assert_eq!(
            prepared.request.messages[0]
                .tool_calls
                .first()
                .map(|tool_call| tool_call.id.as_str()),
            Some("call_2")
        );
        assert_eq!(
            prepared.request.messages[1].tool_call_id.as_deref(),
            Some("call_2")
        );
    }

    #[tokio::test]
    async fn summary_retry_truncates_oversized_chunk() {
        let manager = ContextManager::new(ContextManagerConfig {
            summary_chunk_chars: 10_000,
            preserve_recent_messages: 1,
            min_recent_messages: 1,
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
            summary_overflow_chars: Some(9_500),
        };

        let messages = vec![
            Message::user("older context ".repeat(40_000)),
            Message::assistant("assistant response ".repeat(40_000)),
            Message::user("recent context ".repeat(10_000)),
        ];

        let prepared = manager
            .prepare_request(
                &client,
                ChatRequest::new("claude-sonnet-4-20250514", messages),
            )
            .await
            .unwrap();

        assert!(prepared.compaction.is_some());
        let calls = client.calls.lock().unwrap();
        let summary_calls = calls
            .iter()
            .filter(|request| {
                request
                    .system
                    .as_deref()
                    .unwrap_or_default()
                    .contains("Summarize the earlier conversation context")
            })
            .count();
        assert!(summary_calls >= 2);
    }
}
