//! Validate serving manifests by loading models and checking weight bindings.
use std::env;
use std::path::PathBuf;

use scale_rec::server::registry::ModelRegistry;

fn main() {
    if let Err(err) = run() {
        eprintln!("validate_manifest: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let manifests: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if manifests.is_empty() {
        return Err("usage: validate_manifest <manifest.yaml> [manifest.yaml ...]".into());
    }
    for manifest_path in manifests {
        let parent = manifest_path
            .parent()
            .ok_or_else(|| format!("manifest has no parent: {}", manifest_path.display()))?;
        let registry = ModelRegistry::from_model_dir(parent)?;
        let info = registry.load_manifest(&manifest_path)?;
        println!(
            "validated {} model={} version={}",
            manifest_path.display(),
            info.name,
            info.model_version.unwrap_or_else(|| "default".into())
        );
    }
    Ok(())
}
