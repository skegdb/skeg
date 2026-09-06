use skeg_rigging_net::NetError;

#[test]
fn remote_backpressure_is_retryable() {
    let error = NetError::Remote("BACKPRESSURE memory budget exhausted".into());

    assert!(error.is_retryable());
}

#[test]
fn remote_ratelimited_is_retryable() {
    let error = NetError::Remote("RATELIMITED tenant quota exceeded".into());

    assert!(error.is_retryable());
}

#[test]
fn remote_error_code_prefix_collision_is_not_retryable() {
    let error = NetError::Remote("BACKPRESSUREISH malformed code".into());

    assert!(!error.is_retryable());
}

#[test]
fn protocol_error_is_not_retryable() {
    let error = NetError::Protocol("malformed frame".into());

    assert!(!error.is_retryable());
}
