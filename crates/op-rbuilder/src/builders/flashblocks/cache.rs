use moka::{policy::EvictionPolicy, sync::Cache};
use std::sync::Arc;

use alloy_consensus::transaction::Recovered;
use alloy_eips::eip2718::WithEncoded;
use alloy_primitives::B256;
use op_alloy_rpc_types_engine::OpFlashblockPayload;
use reth_primitives_traits::SignedTransaction;

const MAX_BLOCKS_CACHE_SIZE: u64 = 3;

#[derive(Debug, Clone)]
pub(crate) struct FlashblockPayloadsCache {
    cache: Arc<FlashblockPayloadsCacheInner>,
}

impl FlashblockPayloadsCache {
    pub(crate) fn new(flashblocks_per_block: u64) -> Self {
        Self {
            cache: Arc::new(FlashblockPayloadsCacheInner::new(flashblocks_per_block)),
        }
    }

    pub(crate) fn add_flashblock_payload(&self, payload: OpFlashblockPayload) -> eyre::Result<()> {
        self.cache.add_flashblock_payload(payload)
    }

    pub(crate) fn get_flashblocks_sequence_txs<T: SignedTransaction>(
        &self,
        parent_hash: B256,
    ) -> Option<Vec<WithEncoded<Recovered<T>>>> {
        self.cache.get_flashblocks_sequence_txs(parent_hash)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FlashblockPayloadsCacheInner {
    cache: Cache<B256, Vec<OpFlashblockPayload>>,
    flashblocks_per_block: usize,
}

impl FlashblockPayloadsCacheInner {
    pub(crate) fn new(flashblocks_per_block: u64) -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(MAX_BLOCKS_CACHE_SIZE)
                .eviction_policy(EvictionPolicy::lru())
                .build(),
            flashblocks_per_block: flashblocks_per_block as usize,
        }
    }

    pub(crate) fn add_flashblock_payload(&self, payload: OpFlashblockPayload) -> eyre::Result<()> {
        let parent_hash = payload
            .parent_hash()
            .ok_or_else(|| eyre::eyre!("parent hash in flashblock payload not found"))?;

        self.cache
            .entry_by_ref(&parent_hash)
            .and_upsert_with(|maybe_entry| match maybe_entry {
                Some(entry) => {
                    let mut payloads = entry.into_value();
                    payloads.push(payload);
                    payloads
                }
                None => {
                    let mut payloads = Vec::with_capacity(self.flashblocks_per_block);
                    payloads.push(payload);
                    payloads
                }
            });

        Ok(())
    }

    pub(crate) fn get_flashblocks_sequence_txs<T: SignedTransaction>(
        &self,
        parent_hash: B256,
    ) -> Option<Vec<WithEncoded<Recovered<T>>>> {
        self.cache
            .get(&parent_hash)?
            .iter()
            .flat_map(|payload| payload.recover_transactions())
            .collect::<Result<_, _>>()
            .ok()
    }
}
