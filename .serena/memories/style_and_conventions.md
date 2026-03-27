# Style And Conventions
- Rust naming follows standard conventions: PascalCase for types/traits, snake_case for functions/variables, SCREAMING_SNAKE_CASE for constants.
- Keep imports grouped as std, external crates, then local modules.
- Prefer small shared data types in `src/ai/mod.rs` and provider-specific protocol structs in provider modules.
- Error handling uses custom `AiError` with string payloads and `map_err(|e| AiError::... (e.to_string()))` conversions.
- Async provider interface is exposed through `#[async_trait::async_trait]` and `BoxStream<'static, Result<...>>` for streaming responses.
- Public crate API is re-exported from `src/lib.rs` for straightforward downstream use.