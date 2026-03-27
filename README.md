# conect_llm

複数 LLM provider を共通 API で扱うための Rust ライブラリです。

現時点では以下を提供します。

- 通常の chat / streaming
- provider ごとのデフォルト設定
- Thinking の取得
- provider に応じた Thinking 設定の送信

## 公開 API

外部からは主に以下の型を使います。

- `AiProvider`
- `AiConfig`
- `AiClient`
- `ChatRequest`
- `ChatResponse`
- `Message`
- `StreamChunk`
- `ThinkingConfig`
- `ThinkingOutput`

`src/lib.rs` から再 export されているので、通常は `conect_llm::...` で参照できます。

## 基本的な使い方

```rust
use conect_llm::{
    AiConfig, AiProvider, ChatRequest, Message, ThinkingConfig,
};

let provider = AiProvider::Kimi;
let client = provider.create_client(AiConfig {
    api_key: std::env::var("API_KEY")?,
    base_url: provider.default_base_url().to_string(),
    model: provider.default_model().to_string(),
});

let request = ChatRequest {
    model: client.config().model.clone(),
    messages: vec![
        Message {
            role: "user".to_string(),
            content: "Rust で簡単な HTTP サーバーを書いて".to_string(),
            thinking: None,
        }
    ],
    max_tokens: Some(2048),
    temperature: Some(0.7),
    system: None,
    thinking: Some(ThinkingConfig::enabled()),
};
```

## Thinking は外からどう見えるか

このライブラリでは Thinking を provider 固有の JSON のまま外へ出しません。外部からは共通の Rust 型として見えます。

### 1. 応答全体として取得する

`chat()` の戻り値 `ChatResponse` に `thinking` が入ります。

```rust
let response = client.chat(request).await?;

println!("{}", response.content);

if let Some(thinking) = &response.thinking {
    if let Some(text) = &thinking.text {
        println!("thinking: {}", text);
    }
    if let Some(signature) = &thinking.signature {
        println!("signature: {}", signature);
    }
    if let Some(redacted) = &thinking.redacted {
        println!("redacted: {}", redacted);
    }
}
```

`ThinkingOutput` の意味は次の通りです。

- `text`: 人間が読める thinking 本文または要約
- `signature`: Anthropic 系の署名付き thinking に使う値
- `redacted`: Anthropic 系の `redacted_thinking`

### 2. ストリームとして取得する

`chat_stream()` の各 `StreamChunk` には通常テキストとは別に Thinking 用フィールドがあります。

```rust
use futures_util::StreamExt;

let mut stream = client.chat_stream(request);

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;

    if !chunk.delta.is_empty() {
        print!("{}", chunk.delta);
    }

    if let Some(thinking) = chunk.thinking_delta {
        eprint!("{}", thinking);
    }

    if let Some(signature) = chunk.thinking_signature {
        eprintln!("signature={}", signature);
    }

    if chunk.done {
        break;
    }
}
```

`StreamChunk` の意味は次の通りです。

- `delta`: 通常の本文差分
- `thinking_delta`: Thinking の差分
- `thinking_signature`: Thinking に紐づく署名差分
- `done`: ストリーム完了

### 3. 次のターンへ渡す

Thinking を会話履歴として保持したい場合は、`Message.thinking` にそのまま入れて次の `ChatRequest.messages` へ戻します。

```rust
let assistant_message = Message {
    role: "assistant".to_string(),
    content: response.content.clone(),
    thinking: response.thinking.clone(),
};
```

この形にしておくと、Anthropic 系では `signature` や `redacted` を含む形で再送できます。OpenAI 互換系では `thinking.text` が `reasoning_content` として使われます。

## Thinking の設定

Thinking を要求する側は `ChatRequest.thinking` を使います。

```rust
use conect_llm::{ThinkingConfig, ThinkingDisplay};

let thinking = ThinkingConfig {
    enabled: true,
    budget_tokens: Some(2048),
    display: Some(ThinkingDisplay::Summarized),
    clear_history: None,
};
```

`ThinkingConfig` の意味は次の通りです。

- `enabled`: Thinking を有効にするか
- `budget_tokens`: Anthropic 系の thinking budget
- `display`: Anthropic 系の表示方針
- `clear_history`: Z AI 系の clear thinking 用

簡易ヘルパーもあります。

- `ThinkingConfig::enabled()`
- `ThinkingConfig::disabled()`

ただし、すべての provider がこの設定を受け付けるわけではありません。受け付け可否は `AiProvider::supports_thinking_config()` で判定します。

## Provider 一覧

| Provider | API style | Thinking 出力 | Thinking 設定 |
| --- | --- | --- | --- |
| `AiProvider::Anthropic` | Anthropic | Yes | Yes |
| `AiProvider::GoogleAiStudio` | OpenAI-compatible | No | Yes |
| `AiProvider::Gemini` | Gemini native | Yes | Yes |
| `AiProvider::OpenAi` | OpenAI-compatible | No | No |
| `AiProvider::Sakura` | OpenAI-compatible | Yes | No |
| `AiProvider::Kimi` | OpenAI-compatible | Yes | Yes |
| `AiProvider::KimiCoding` | Anthropic | Yes | Yes |
| `AiProvider::ZAi` | OpenAI-compatible | Yes | Yes |
| `AiProvider::ZAiCoding` | OpenAI-compatible | Yes | Yes |

各 provider からは以下も取得できます。

- `name()`
- `default_base_url()`
- `default_model()`
- `supports_thinking_output()`
- `supports_thinking_config()`

```rust
let provider = AiProvider::Sakura;

println!("{}", provider.name());
println!("{}", provider.default_base_url());
println!("{}", provider.default_model());
println!("{}", provider.supports_thinking_output());
println!("{}", provider.supports_thinking_config());
```

## 注意点

- `AiProvider::GoogleAiStudio` は `https://generativelanguage.googleapis.com/v1beta/openai` の OpenAI compatibility を使います。
- `AiProvider::Gemini` は `generateContent` / `streamGenerateContent` を使う native Gemini API です。
- `AiProvider::OpenAi` は現状 `chat.completions` ベースです。Thinking は公開 capability としては `false` 扱いです。
- `AiProvider::GoogleAiStudio` は request 側で Gemini の `thinking_config` を送れますが、このライブラリでは structured thinking output の公開 capability は `false` にしています。
- OpenAI 互換 provider でも `reasoning_content` を返す実装なら、transport 側は受け取れるようにしてあります。
- `Message.thinking` は provider によっては一部しか使われません。OpenAI 互換系では主に `text` を再送します。
- `chat_stream()` は `BoxStream<'static, Result<StreamChunk, AiError>>` を返します。

## 開発

整形と確認:

```bash
cargo fmt
cargo check
cargo test
```
