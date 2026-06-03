use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();
static VERBOSE: OnceLock<bool> = OnceLock::new();

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| env_flag("SCALE_REC_PROFILE_UNIMIXER"))
}

pub fn verbose() -> bool {
    *VERBOSE.get_or_init(|| env_flag("SCALE_REC_PROFILE_UNIMIXER_VERBOSE"))
}

pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub fn log(stage: &str, start: Option<Instant>) {
    if let Some(start) = start {
        eprintln!(
            "[unimixer-profile] {stage}: {:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}
