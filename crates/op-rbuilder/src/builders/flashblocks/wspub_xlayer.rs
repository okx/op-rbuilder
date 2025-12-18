use op_alloy_rpc_types_engine::OpFlashblockPayload;
use std::io;
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Utf8Bytes;
use tracing::debug;

pub(super) fn publish_op_payload(
    pipe: &broadcast::Sender<Utf8Bytes>,
    payload: &OpFlashblockPayload,
) -> io::Result<usize> {
    debug!(
        target: "payload_builder",
        message = "Sending OpFlashblockPayload",
        payload_id = payload.payload_id.to_string(),
        index = payload.index,
        base = payload.base.is_some(),
    );

    let serialized = serde_json::to_string(payload)?;
    let utf8_bytes = Utf8Bytes::from(serialized);
    let size = utf8_bytes.len();
    // Send the serialized payload to all subscribers
    pipe.send(utf8_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::ConnectionAborted, e))?;
    Ok(size)
}
