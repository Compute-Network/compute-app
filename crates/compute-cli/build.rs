use std::process::Command;

fn main() {
    // Embed git hash
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());

    println!("cargo:rustc-env=COMPUTE_GIT_HASH={git_hash}");

    // Embed build date
    let build_date = chrono_lite_date();
    println!("cargo:rustc-env=COMPUTE_BUILD_DATE={build_date}");

    // Re-run if git HEAD changes
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}

fn chrono_lite_date() -> String {
    // Simple date without external dep

    Command::new("date")
        .args(["+%Y-%m-%d"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}
