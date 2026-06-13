//! 性能分析工具：通过环境变量控制 UniMixer 各阶段耗时埋点。
use std::sync::OnceLock;
use std::time::Instant;
use tracing::info;

static ENABLED: OnceLock<bool> = OnceLock::new();
static VERBOSE: OnceLock<bool> = OnceLock::new();

/// 性能分析是否启用 (SCALE_REC_PROFILE_UNIMIXER)。
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| env_flag("SCALE_REC_PROFILE_UNIMIXER"))
}

/// 详细模式是否启用 (SCALE_REC_PROFILE_UNIMIXER_VERBOSE)。
pub fn verbose() -> bool {
    *VERBOSE.get_or_init(|| env_flag("SCALE_REC_PROFILE_UNIMIXER_VERBOSE"))
}

/// 若启用则返回 Instant::now()。
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// 若启用以 info 级别记录阶段耗时。
pub fn log(stage: &str, start: Option<Instant>) {
    if let Some(start) = start {
        info!(
            target = "unimixer-profile",
            stage = %stage,
            elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
            "stage timing"
        );
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}
