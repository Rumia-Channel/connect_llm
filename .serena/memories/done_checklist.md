# Done Checklist
Before finishing a code change in this project:
- Run `cargo fmt`.
- Run `cargo check`.
- Run `cargo test` when feasible.
- Confirm the public API in `src/lib.rs` still matches the intended downstream usage.
- If the change affects provider behavior, check both OpenAI-compatible and Anthropic-compatible paths.