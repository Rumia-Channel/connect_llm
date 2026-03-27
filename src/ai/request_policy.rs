pub(crate) fn normalized_model_id(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    let model = normalized_model_id(model);
    model.starts_with("gpt-5")
        || model.contains("codex")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("o5")
}

pub(crate) fn sanitize_openai_style_temperature(
    base_url: &str,
    model: &str,
    temperature: Option<f32>,
) -> Option<f32> {
    if temperature.is_none() {
        return None;
    }

    let base_url = base_url.trim_end_matches('/').to_ascii_lowercase();
    let uses_openai_reasoning_controls = !base_url.contains("moonshot.ai")
        && !base_url.contains("api.z.ai")
        && !base_url.contains("generativelanguage.googleapis.com")
        && !base_url.contains("api.ai.sakura.ad.jp");

    if uses_openai_reasoning_controls && is_openai_reasoning_model(model) {
        None
    } else {
        temperature
    }
}

pub(crate) fn sanitize_codex_temperature(_model: &str, _temperature: Option<f32>) -> Option<f32> {
    None
}

pub(crate) fn sanitize_github_copilot_temperature(
    model: &str,
    temperature: Option<f32>,
) -> Option<f32> {
    if is_openai_reasoning_model(model) {
        None
    } else {
        temperature
    }
}

pub(crate) fn sanitize_github_copilot_reasoning_effort(
    model: &str,
    reasoning_effort: Option<&'static str>,
) -> Option<&'static str> {
    let model = normalized_model_id(model);
    if model.contains("claude") || model.contains("gemini") {
        None
    } else {
        reasoning_effort
    }
}

pub(crate) fn sanitize_github_copilot_thinking_budget(
    model: &str,
    thinking_budget: Option<u32>,
) -> Option<u32> {
    let model = normalized_model_id(model);
    if model.contains("claude") {
        thinking_budget
    } else {
        None
    }
}
