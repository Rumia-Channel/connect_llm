# conect_llm

複数 LLM provider を共通 API で扱うための Rust ライブラリです。

現時点では以下を提供します。

- 通常の chat / streaming
- provider ごとのデフォルト設定
- Thinking の取得
- provider に応じた Thinking 設定の送信
- Tool Use の定義送信と tool call / tool result の送受信

## 公開 API

外部からは主に以下の型を使います。

- `AiProvider`
- `AiConfig`
- `AiClient`
- `ChatRequest`
- `ChatResponse`
- `DebugTrace`
- `Message`
- `StreamChunk`
- `ThinkingConfig`
- `ThinkingEffort`
- `ThinkingOutput`
- `ToolDefinition`
- `ToolChoice`
- `ToolCall`
- `ToolCallDelta`

`src/lib.rs` から再 export されているので、通常は `conect_llm::...` で参照できます。

## デバッグ

ライブラリ全体の debug flag は `set_debug_logging(true)` で有効にできます。

```rust
use conect_llm::set_debug_logging;

set_debug_logging(true);
```

有効化すると次の 2 つが起きます。

- stderr に provider ごとの raw request / raw response / raw SSE が出る
- `ChatResponse.debug` と `StreamChunk.debug` に raw payload が入る

`DebugTrace` は共通で次の形です。

- `request`: 生の request body
- `response`: 生の response body または stream event

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

let mut request = ChatRequest::new(
    client.config().model.clone(),
    vec![Message::user("Rust で簡単な HTTP サーバーを書いて")],
);
request.max_tokens = Some(2048);
request.temperature = Some(0.7);
request.thinking = Some(ThinkingConfig::enabled());
```

`Message::user(...)`, `Message::assistant(...)`, `Message::assistant_tool_calls(...)`, `Message::tool_result(...)`, `ChatRequest::new(...)`, `ToolDefinition::function(...)` を用意しているので、通常は field を全部手で埋める必要はありません。

## Thinking は外からどう見えるか

このライブラリでは Thinking を provider 固有の JSON のまま外へ出しません。外部からは共通の Rust 型として見えます。

### 1. 応答全体として取得する

`chat()` の戻り値 `ChatResponse` に `thinking` が入ります。

```rust
let response = client.chat(request).await?;

println!("{}", response.content);

if let Some(debug) = &response.debug {
    if let Some(request) = &debug.request {
        eprintln!("raw request: {}", request);
    }
    if let Some(response) = &debug.response {
        eprintln!("raw response: {}", response);
    }
}

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
- `signature`: Anthropic 系の署名付き thinking や GitHub Copilot の `reasoning_opaque` に使う値
- `redacted`: Anthropic 系の `redacted_thinking`

## Tool Use

Tool Use は provider 固有の request shape を隠して、共通の型で扱います。

### 1. tool 定義を送る

```rust
use conect_llm::{ChatRequest, Message, ToolChoice, ToolDefinition};
use serde_json::json;

let mut request = ChatRequest::new(
    "gpt-5.4",
    vec![Message::user("東京の天気を調べて")],
);
request.tools = vec![
    ToolDefinition::function(
        "get_weather",
        Some("都市名から現在の天気を返す".to_string()),
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    ),
];
request.tool_choice = Some(ToolChoice::Auto);
```

### 2. LLM からの tool call を受ける

non-stream では `ChatResponse.tool_calls` に入ります。

```rust
let response = client.chat(request).await?;

for tool_call in &response.tool_calls {
    println!("tool={} args={}", tool_call.name, tool_call.arguments);
}
```

`ToolCall` の意味は次の通りです。

- `id`: provider 側の tool call id
- `name`: 呼び出す tool 名
- `arguments`: JSON 引数

stream では `StreamChunk.tool_call_deltas` に差分が入ります。

```rust
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;

    for delta in chunk.tool_call_deltas {
        println!(
            "tool delta index={} id={:?} name={:?} args={:?}",
            delta.index, delta.id, delta.name, delta.arguments
        );
    }
}
```

`ToolCallDelta` の意味は次の通りです。

- `index`: 同一レスポンス内の call 順
- `id`: call id の差分または確定値
- `name`: tool 名の差分または確定値
- `arguments`: 引数 JSON の差分

### 3. tool result を返す

次ターンでは `Message::assistant_tool_calls(...)` と `Message::tool_result(...)` を会話履歴へ戻します。

```rust
use conect_llm::{Message, ToolCall};
use serde_json::json;

let tool_call = ToolCall {
    id: "call_123".to_string(),
    name: "get_weather".to_string(),
    arguments: json!({ "city": "東京" }),
};

let messages = vec![
    Message::user("東京の天気を調べて"),
    Message::assistant_tool_calls(vec![tool_call.clone()]),
    Message::tool_result(
        tool_call.id.clone(),
        tool_call.name.clone(),
        json!({ "temperature_c": 22, "condition": "sunny" }),
    ),
];
```

tool result は `tool_result: serde_json::Value` として保持します。OpenAI 互換系では文字列化し、Anthropic / Gemini では provider native の tool result block へ変換します。

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

    if let Some(debug) = &chunk.debug {
        if let Some(request) = &debug.request {
            eprintln!("raw request: {}", request);
        }
        if let Some(response) = &debug.response {
            eprintln!("raw event: {}", response);
        }
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
- `tool_call_deltas`: Tool call の差分
- `done`: ストリーム完了
- `debug`: debug flag 有効時の raw request / raw event

### 3. 次のターンへ渡す

Thinking を会話履歴として保持したい場合は、`Message.thinking` にそのまま入れて次の `ChatRequest.messages` へ戻します。

```rust
let mut assistant_message = Message::assistant(response.content.clone());
assistant_message.thinking = response.thinking.clone();
assistant_message.tool_calls = response.tool_calls.clone();
```

この形にしておくと、Anthropic 系では `signature` や `redacted` を含む形で再送できます。OpenAI 互換系では `thinking.text` が `reasoning_content` として使われる想定ですが、provider によっては Sakura のように `reasoning` フィールドで返す実装もあるため、このライブラリ側で吸収しています。

## Thinking の設定

Thinking を要求する側は `ChatRequest.thinking` を使います。

```rust
use conect_llm::{ThinkingConfig, ThinkingDisplay, ThinkingEffort};

let thinking = ThinkingConfig {
    enabled: true,
    effort: Some(ThinkingEffort::High),
    budget_tokens: Some(2048),
    display: Some(ThinkingDisplay::Summarized),
    clear_history: None,
};
```

`ThinkingConfig` の意味は次の通りです。

- `enabled`: Thinking を有効にするか
- `effort`: OpenAI Codex などで使う思考レベル
- `budget_tokens`: Anthropic 系の thinking budget
- `display`: Anthropic 系の表示方針
- `clear_history`: Z AI 系の clear thinking 用

`ThinkingEffort` は次を持ちます。

- `Minimal`
- `Low`
- `Medium`
- `High`
- `XHigh`

簡易ヘルパーもあります。

- `ThinkingConfig::enabled()`
- `ThinkingConfig::enabled_with_effort()`
- `ThinkingConfig::disabled()`

`ThinkingConfig::enabled()` は共通の「thinking を出す」設定です。Codex の思考レベルを変えたいときだけ `enabled_with_effort()` か、`effort: Some(...)` を使います。

ただし、すべての provider が request 側の設定を受け付けるわけではありません。受け付け可否は `AiProvider::supports_thinking_config()` で判定します。Sakura のように request 側では指定できなくても、レスポンス側で `reasoning_content` や `reasoning` を返す provider はあります。

## Provider 一覧

| Provider | API style | Thinking 出力 | Thinking 設定 | Tool Use |
| --- | --- | --- | --- | --- |
| `AiProvider::Anthropic` | Anthropic | Yes | Yes | Yes |
| `AiProvider::GitHubCopilot` | OpenAI-compatible (Copilot) | Yes | Yes | Yes |
| `AiProvider::GoogleAiStudio` | OpenAI-compatible | No | Yes | Yes |
| `AiProvider::Gemini` | Gemini native | Yes | Yes | Yes |
| `AiProvider::OpenAi` | OpenAI-compatible | No | No | Yes |
| `AiProvider::OpenAiCodex` | ChatGPT Codex backend | Yes | Yes | Yes |
| `AiProvider::Sakura` | OpenAI-compatible | Yes | No | Yes |
| `AiProvider::Kimi` | OpenAI-compatible | Yes | Yes | Yes |
| `AiProvider::KimiCoding` | Anthropic | Yes | Yes | Yes |
| `AiProvider::ZAi` | OpenAI-compatible | Yes | Yes | Yes |
| `AiProvider::ZAiCoding` | OpenAI-compatible | Yes | Yes | Yes |

各 provider からは以下も取得できます。

- `name()`
- `default_base_url()`
- `default_model()`
- `supports_thinking_output()`
- `supports_thinking_config()`
- `supports_tools()`

```rust
let provider = AiProvider::Sakura;

println!("{}", provider.name());
println!("{}", provider.default_base_url());
println!("{}", provider.default_model());
println!("{}", provider.supports_thinking_output());
println!("{}", provider.supports_thinking_config());
println!("{}", provider.supports_tools());
```

## 注意点

- `AiProvider::GoogleAiStudio` は `https://generativelanguage.googleapis.com/v1beta/openai` の OpenAI compatibility を使います。
- `AiProvider::Gemini` は `generateContent` / `streamGenerateContent` を使う native Gemini API です。
- `AiProvider::GitHubCopilot` は `chat/completions` を使う Copilot の OpenAI-compatible endpoint を叩きます。
- `AiProvider::GitHubCopilot` は request header に `Openai-Intent: conversation-edits` と `x-initiator` を付けます。ここは `opencode` の実装に合わせています。
- `AiProvider::GitHubCopilot` の `api_key` は GitHub token として扱い、内部で Copilot API token に交換します。
- `AiProvider::GitHubCopilot` では `api_key` を空にすると `COPILOT_HOME/auth.json` または `~/.copilot/auth.json` から GitHub token を読みます。
- `AiProvider::GitHubCopilot` は `ChatRequest.thinking.effort` を `reasoning_effort` として、`budget_tokens` を `thinking_budget` として送ります。
- `AiProvider::OpenAi` は現状 `chat.completions` ベースです。Thinking は公開 capability としては `false` 扱いです。
- `AiProvider::OpenAiCodex` は `https://chatgpt.com/backend-api/codex/responses` を使います。
- `AiProvider::OpenAiCodex` は OpenAI Responses item 互換の `function_call` / `function_call_output` として Tool Use を扱います。
- `AiProvider::OpenAiCodex` の `api_key` は通常の OpenAI API key ではなく、ChatGPT OAuth の access token として扱います。
- `AiProvider::OpenAiCodex` では `api_key` を空にすると `CODEX_HOME/auth.json` または `~/.codex/auth.json` から access token / refresh token を読みます。
- `AiProvider::OpenAiCodex` は access token の期限が近い場合、`refresh_token` を使って `auth.openai.com/oauth/token` で更新し、`auth.json` へ書き戻します。
- `AiProvider::OpenAiCodex` では `ChatRequest.model` でモデルを選べます。
- `AiProvider::OpenAiCodex` では `ChatRequest.thinking.effort` を `reasoning.effort` として送ります。`ThinkingConfig::enabled()` は thinking を有効にするだけで、effort を明示しない場合は Codex 側のデフォルトに委ねます。
- `AiProvider::GoogleAiStudio` は request 側で Gemini の `thinking_config` を送れますが、このライブラリでは structured thinking output の公開 capability は `false` にしています。
- OpenAI 互換 provider でも `reasoning_content` または `reasoning` を返す実装なら、transport 側は受け取れるようにしてあります。
- `Message.thinking` は provider によっては一部しか使われません。OpenAI 互換系では主に `text` を再送します。
- `chat_stream()` は `BoxStream<'static, Result<StreamChunk, AiError>>` を返します。

## OpenAI Codex の使い方

`opencode` に寄せて、ChatGPT OAuth ベースの Codex backend を直接叩けるようにしています。

### ブラウザからログインする

```rust
use conect_llm::{
    login_openai_codex_via_browser, OpenAiCodexBrowserAuthOptions,
};

let auth = login_openai_codex_via_browser(OpenAiCodexBrowserAuthOptions::default())?;

println!("{}", auth.auth_path.display());
```

この helper は以下を行います。

- `localhost:1455` で callback を待ち受ける
- ブラウザを開いて ChatGPT OAuth を開始する
- `auth.openai.com/oauth/token` で code exchange を行う
- `~/.codex/auth.json` または `CODEX_HOME/auth.json` に保存する

`opencode` と同様、保存先の `auth.json` は後続の `AiProvider::OpenAiCodex` からそのまま利用できます。

sample CLI で `AiProvider::OpenAiCodex` を選び、`API key` を空のまま進めた場合も、保存済み `auth.json` がなければ同じ browser login を自動で開始します。CLI 側では `spawn_blocking` 経由で呼ぶため、Tokio runtime 上での blocking client panic を避けています。

### Codex CLI のログイン情報をそのまま使う

```rust
use conect_llm::{AiConfig, AiProvider};

let provider = AiProvider::OpenAiCodex;
let client = provider.create_client(AiConfig {
    api_key: String::new(),
    base_url: provider.default_base_url().to_string(),
    model: provider.default_model().to_string(),
});
```

この場合は `~/.codex/auth.json` を読みます。`CODEX_HOME` を設定している場合はそちらが優先されます。

### access token を直接渡す

```rust
use conect_llm::{AiConfig, AiProvider};

let provider = AiProvider::OpenAiCodex;
let client = provider.create_client(AiConfig {
    api_key: std::env::var("OPENAI_OAUTH_ACCESS_TOKEN")?,
    base_url: provider.default_base_url().to_string(),
    model: "gpt-5.1-codex-max".to_string(),
});
```

この場合、`api_key` は `sk-...` の API key ではなく ChatGPT OAuth の bearer token です。

### モデルと思考レベルを指定する

```rust
use conect_llm::{
    AiConfig, AiProvider, ChatRequest, Message, ThinkingConfig, ThinkingEffort,
};

let provider = AiProvider::OpenAiCodex;
let client = provider.create_client(AiConfig {
    api_key: String::new(),
    base_url: provider.default_base_url().to_string(),
    model: "gpt-5.1-codex-mini".to_string(),
});

let mut request = ChatRequest::new(
    "gpt-5.1-codex-max",
    vec![Message::user("この変更方針で進めて")],
);
request.max_tokens = Some(8192);
request.thinking = Some(ThinkingConfig::enabled_with_effort(ThinkingEffort::XHigh));
```

Codex ではこの `thinking.effort` を `reasoning.effort` として送ります。thinking の ON/OFF 自体は `ThinkingConfig::enabled()` / `ThinkingConfig::disabled()` で切り替え、Codex の強さだけ変えたいときに `effort` を足す形です。モデルは `AiConfig.model` と `ChatRequest.model` のどちらでも指定できますが、実際に送信されるのは `ChatRequest.model` です。

## GitHub Copilot の使い方

`opencode` の device flow を踏襲して、GitHub token を取得し、それを Copilot API token に交換して使う形にしています。

### device login を実行する

```rust
use conect_llm::{
    login_github_copilot_via_device, GitHubCopilotDeviceAuthOptions,
};

let auth = login_github_copilot_via_device(GitHubCopilotDeviceAuthOptions::default())?;

println!("{}", auth.auth_path.display());
```

この helper は以下を行います。

- GitHub device code を発行する
- URL と one-time code を表示する
- 認可完了まで polling する
- `~/.copilot/auth.json` または `COPILOT_HOME/auth.json` に保存する

### 保存済み auth をそのまま使う

```rust
use conect_llm::{AiConfig, AiProvider};

let provider = AiProvider::GitHubCopilot;
let client = provider.create_client(AiConfig {
    api_key: String::new(),
    base_url: provider.default_base_url().to_string(),
    model: provider.default_model().to_string(),
});
```

この場合は保存済みの GitHub token を読み、内部で Copilot API token へ交換します。

### GitHub token を直接渡す

```rust
use conect_llm::{AiConfig, AiProvider};

let provider = AiProvider::GitHubCopilot;
let client = provider.create_client(AiConfig {
    api_key: std::env::var("GITHUB_TOKEN")?,
    base_url: provider.default_base_url().to_string(),
    model: "claude-sonnet-4.5".to_string(),
});
```

### thinking を再送する

GitHub Copilot は visible thinking とは別に `reasoning_opaque` を返すことがあります。このライブラリでは `ThinkingOutput.signature` に入れて再送します。

## 開発

整形と確認:

```bash
cargo fmt
cargo check
cargo test
```
