//! Optional OpenTelemetry OTLP tracing bridge.
//!
//! Gated behind the `tracing-otlp` cargo feature so the default build
//! does not link the `opentelemetry*` crates. When the feature is on
//! AND the env var `SKEG_TRACE_OTLP_ENDPOINT` is set, this module
//! returns a `tracing_subscriber::Layer` that ships spans to the
//! configured OTLP/gRPC collector.
//!
//! Sampling: head-based. `SKEG_TRACE_SAMPLE_RATE=0.05` keeps 5% of
//! traces. Default 1.0 (sample all). The sampler runs once at span
//! creation, so the cost on non-sampled spans is one rng draw.

#![cfg(feature = "tracing-otlp")]

use std::str::FromStr;
use std::sync::OnceLock;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use tracing::Subscriber;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::registry::LookupSpan;

/// Stash the provider so [`shutdown`] can drain its batch queue on
/// clean exit. `OnceLock` rejects re-installation, which keeps the
/// lifecycle obvious if some test ends up calling `install` twice.
static PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

const SERVICE_NAME: &str = "skeg";
const TRACER_NAME: &str = "skeg-server";
const ENV_ENDPOINT: &str = "SKEG_TRACE_OTLP_ENDPOINT";
const ENV_SAMPLE_RATE: &str = "SKEG_TRACE_SAMPLE_RATE";
const ENV_RESOURCE_ATTRS: &str = "SKEG_TRACE_RESOURCE_ATTRS";

/// If `SKEG_TRACE_OTLP_ENDPOINT` is set, build an OTLP/gRPC exporter,
/// install a batched span processor, and return a `tracing-opentelemetry`
/// layer ready to be `.with()`-chained onto the subscriber registry.
///
/// Returns `None` (and logs a `tracing::info`) when the env var is
/// absent so the caller can chain `.with(install(...))` unconditionally.
///
/// # Errors
///
/// Returns `Err` only when the env var IS set but the exporter cannot
/// be built (malformed endpoint, gRPC channel setup failure, etc).
/// The caller propagates this to `main` so the binary refuses to start
/// silently mis-configured.
pub fn install<S>()
-> Result<Option<OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>, OtlpError>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let Ok(endpoint) = std::env::var(ENV_ENDPOINT) else {
        return Ok(None);
    };
    if endpoint.is_empty() {
        return Ok(None);
    }

    let sample_rate = sample_rate_from_env();

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()
        .map_err(OtlpError::Exporter)?;

    let resource = build_resource();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(Sampler::TraceIdRatioBased(sample_rate))
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(TRACER_NAME);

    // Install globally so libraries that bypass the tracing bridge can
    // still observe the same tracer, and stash a clone for shutdown.
    opentelemetry::global::set_tracer_provider(provider.clone());
    let _ = PROVIDER.set(provider);

    tracing::info!(
        target: "skeg::tracing_otlp",
        endpoint = %endpoint,
        sample_rate,
        "OTLP/gRPC tracing exporter installed"
    );

    Ok(Some(tracing_opentelemetry::layer().with_tracer(tracer)))
}

/// Shut the exporter down on a clean exit so in-flight spans flush to
/// the collector. Safe to call when the exporter was never installed.
pub fn shutdown() {
    if let Some(p) = PROVIDER.get()
        && let Err(e) = p.shutdown()
    {
        // Don't propagate; the binary is on its way out and the
        // collector either gets the batch or doesn't.
        tracing::warn!(target: "skeg::tracing_otlp", error = %e, "OTLP shutdown failed");
    }
}

fn sample_rate_from_env() -> f64 {
    let Ok(raw) = std::env::var(ENV_SAMPLE_RATE) else {
        return 1.0;
    };
    f64::from_str(&raw).map_or(1.0, |v| v.clamp(0.0, 1.0))
}

fn build_resource() -> Resource {
    let mut builder = Resource::builder()
        .with_service_name(SERVICE_NAME)
        .with_attribute(opentelemetry::KeyValue::new(
            "service.version",
            env!("CARGO_PKG_VERSION"),
        ));

    // `SKEG_TRACE_RESOURCE_ATTRS=key1=v1,key2=v2` lets operators inject
    // per-deployment labels (host, region, tenant) without recompiling.
    if let Ok(raw) = std::env::var(ENV_RESOURCE_ATTRS) {
        for pair in raw.split(',') {
            if let Some((k, v)) = pair.split_once('=') {
                let k = k.trim().to_owned();
                let v = v.trim().to_owned();
                if !k.is_empty() && !v.is_empty() {
                    builder = builder.with_attribute(opentelemetry::KeyValue::new(k, v));
                }
            }
        }
    }

    builder.build()
}

/// Failure modes from [`install`].
#[derive(Debug, thiserror::Error)]
pub enum OtlpError {
    #[error("failed to build OTLP exporter: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),
}
