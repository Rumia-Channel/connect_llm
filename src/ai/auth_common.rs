use crate::ai::AiError;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

pub(crate) fn open_url_in_browser(url: &str) -> Result<(), AiError> {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(|error| {
                AiError::Api(format!(
                    "Failed to open browser automatically: {}. Open this URL manually: {}",
                    error, url
                ))
            })?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn().map_err(|error| {
            AiError::Api(format!(
                "Failed to open browser automatically: {}. Open this URL manually: {}",
                error, url
            ))
        })?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(url).spawn().map_err(|error| {
            AiError::Api(format!(
                "Failed to open browser automatically: {}. Open this URL manually: {}",
                error, url
            ))
        })?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err(AiError::Api(format!(
        "Automatic browser opening is not supported on this platform. Open this URL manually: {}",
        url
    )))
}
