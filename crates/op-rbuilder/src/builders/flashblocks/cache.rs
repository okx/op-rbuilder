use parking_lot::Mutex;
use std::sync::Arc;

use alloy_consensus::transaction::Recovered;
use alloy_eips::eip2718::WithEncoded;
use alloy_primitives::B256;
use op_alloy_rpc_types_engine::OpFlashblockPayload;
use reth_payload_builder::PayloadId;
use reth_primitives_traits::SignedTransaction;

type FlashblockPayloadsSequence = Option<(PayloadId, Option<B256>, Vec<OpFlashblockPayload>)>;

/// Cache for the current pending block's flashblock payloads sequence that is
/// being built, based on the `payload_id`.
#[derive(Debug, Clone, Default)]
pub(crate) struct FlashblockPayloadsCache {
    inner: Arc<Mutex<FlashblockPayloadsSequence>>,
}

impl FlashblockPayloadsCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add_flashblock_payload(&self, payload: OpFlashblockPayload) -> eyre::Result<()> {
        let mut guard = self.inner.lock();
        match guard.as_mut() {
            Some((curr_payload_id, parent_hash, payloads))
                if *curr_payload_id == payload.payload_id =>
            {
                if parent_hash.is_none()
                    && let Some(hash) = payload.parent_hash()
                {
                    *parent_hash = Some(hash);
                }
                payloads.push(payload);
            }
            _ => {
                // New payload_id - replace entire cache
                *guard = Some((payload.payload_id, payload.parent_hash(), vec![payload]));
            }
        }
        Ok(())
    }

    /// Get the flashblocks sequence transactions for a given `parent_hash`. Note that we do not
    /// yield sequencer transactions that were included in the payload attributes (index 0).
    ///
    /// Returns `None` if:
    /// - `parent_hash` is not the current pending block's parent hash
    /// - The payloads are not in sequential order or have missing indexes
    pub(crate) fn get_flashblocks_sequence_txs<T: SignedTransaction>(
        &self,
        parent_hash: B256,
    ) -> Option<Vec<WithEncoded<Recovered<T>>>> {
        let mut payloads = {
            let mut guard = self.inner.lock();
            let cache_ref = match guard.as_ref() {
                Some(r) => r,
                None => {
                    tracing::info!(
                        target: "payload_builder",
                        ?parent_hash,
                        "[DEBUG] p2p flashblocks cache is empty, no payloads to replay"
                    );
                    return None;
                }
            };
            let (cached_payload_id, curr_parent_hash, cached_payloads) = cache_ref;
            if *curr_parent_hash != Some(parent_hash) {
                tracing::info!(
                    target: "payload_builder",
                    ?parent_hash,
                    ?curr_parent_hash,
                    ?cached_payload_id,
                    cached_payload_count = cached_payloads.len(),
                    "[DEBUG] p2p flashblocks cache parent hash mismatch"
                );
                return None;
            }
            // Take ownership and flush the cache
            let (_, _, payloads) = guard.take()?;
            payloads
        };

        payloads.sort_by_key(|p| p.index);

        let payload_count = payloads.len();
        let indices: Vec<u64> = payloads.iter().map(|p| p.index).collect();
        tracing::info!(
            target: "payload_builder",
            ?parent_hash,
            payload_count,
            ?indices,
            "[DEBUG] p2p flashblocks cache hit, attempting transaction recovery"
        );

        // Skip base payload index 0 (sequencer transactions)
        payloads.iter().skip(1).enumerate().try_fold(
            Vec::with_capacity(payloads.len()),
            |mut acc, (expected_index, payload)| {
                if payload.index != expected_index as u64 + 1 {
                    tracing::warn!(
                        target: "payload_builder",
                        expected = expected_index + 1,
                        got = payload.index,
                        ?parent_hash,
                        "[DEBUG] flashblock payloads have missing or out-of-order indexes"
                    );
                    return None;
                }
                match payload
                    .recover_transactions()
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(txs) => {
                        acc.extend(txs);
                        Some(acc)
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "payload_builder",
                            index = payload.index,
                            ?parent_hash,
                            error = %e,
                            "[DEBUG] failed to recover transactions from cached flashblock payload"
                        );
                        None
                    }
                }
            },
        )
    }
}
