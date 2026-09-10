// Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// MindIE is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

//! KV Conductor — Standalone KV cache indexer service for MindIE-PyMotor.
//!
//! Starts an HTTP server that maintains radix-tree-based KV cache indexes
//! per (model, tenant) pair, answering overlap queries to guide cache-aware
//! request routing decisions.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::fmt::time::OffsetTime;
use tracing_subscriber::EnvFilter;

use kv_conductor::indexer::CacheMaintenanceConfig;
use kv_conductor::registry::WorkerRegistry;
use kv_conductor::server::{create_router, AppState};

/// KV Conductor — Radix-tree-based KV cache indexer for MindIE-PyMotor.
#[derive(Parser, Debug)]
#[command(name = "kv-conductor")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Host address to bind to (IPv4/IPv6, dual-stack by default)
    #[arg(long, default_value = "::")]
    host: String,

    /// Port to listen on
    #[arg(long, short, default_value = "13333")]
    port: u16,

    /// Maintenance sweep interval in seconds.
    #[arg(long, default_value = "30")]
    maintenance_interval_secs: u64,

    /// Maximum age of a pending pool entry in seconds.
    #[arg(long, default_value = "60")]
    pending_ttl_secs: u64,

    /// Maximum age of a retained content entry in seconds.
    #[arg(long, default_value = "300")]
    content_ttl_secs: u64,

    /// Maximum age of an unmatched offload entry in seconds.
    #[arg(long, default_value = "600")]
    offload_ttl_secs: u64,
}

#[tokio::main]
async fn main() {
    // Initialize tracing. Default `info` hides TRACE/DEBUG kv_event ingest
    // spam. HTTP access/connection traces from hyper/tower-http are always
    // capped at info unless the operator names those targets in RUST_LOG.
    tracing_subscriber::fmt()
        .with_env_filter(build_env_filter())
        .with_target(false)
        .with_timer(OffsetTime::new(
            time::UtcOffset::from_hms(8, 0, 0).expect("invalid UTC+8 offset"),
            time::format_description::well_known::Rfc3339,
        ))
        .init();

    let cli = Cli::parse();

    let host: IpAddr = cli.host.parse().expect("invalid host address");
    let addr = SocketAddr::new(host, cli.port);

    let registry = Arc::new(WorkerRegistry::with_cache_config(CacheMaintenanceConfig {
        pending_ttl: std::time::Duration::from_secs(cli.pending_ttl_secs),
        content_ttl: std::time::Duration::from_secs(cli.content_ttl_secs),
        offload_ttl: std::time::Duration::from_secs(cli.offload_ttl_secs),
    }));
    let maintenance_registry = Arc::downgrade(&registry);
    let maintenance_interval = std::time::Duration::from_secs(cli.maintenance_interval_secs.max(1));
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(maintenance_interval);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let Some(registry) = maintenance_registry.upgrade() else {
                break;
            };
            let pruned = registry.maintenance().await;
            if pruned > 0 {
                tracing::debug!(pruned, "cache maintenance completed");
            }
        }
    });
    let state = AppState { registry };
    let router = create_router(state);

    tracing::info!(
        rust_log = %std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        "KV conductor starting on {}",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind TCP listener");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    tracing::info!("KV conductor shut down");
}

/// Build the tracing filter from `RUST_LOG`, defaulting to `info`.
///
/// Caps `hyper` / `tower_http` at info unless the operator names them, so
/// leftover `RUST_LOG=trace` (ConfigMap overwriting the Pod spec) does not
/// flood stdout with `TRACE connection accepted/closed` and `DEBUG request`.
fn build_env_filter() -> EnvFilter {
    let mut spec = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let lowered = spec.to_ascii_lowercase();
    if !directive_mentions(&lowered, "tower_http") {
        spec.push_str(",tower_http=info");
    }
    if !directive_mentions(&lowered, "hyper") {
        spec.push_str(",hyper=info");
    }
    EnvFilter::try_new(&spec).unwrap_or_else(|_| EnvFilter::new("info"))
}

fn directive_mentions(spec: &str, crate_name: &str) -> bool {
    spec.split(',').any(|directive| {
        let name = directive.split('=').next().unwrap_or("").trim();
        name == crate_name || name.starts_with(&format!("{crate_name}::"))
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let sigterm = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(unix)]
    tokio::select! {
        () = ctrl_c => {},
        () = sigterm => {},
    }

    #[cfg(not(unix))]
    ctrl_c.await;

    tracing::info!("received shutdown signal, draining connections...");
}
