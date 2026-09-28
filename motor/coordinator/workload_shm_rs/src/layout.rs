// Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// MindIE is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

//! Byte layout for the workload shared-memory segment.
//!
//! This mirrors, byte-for-byte, the Python layout in
//! `motor/coordinator/scheduler/runtime/workload_shm/layout.py` (SCHEMA_VERSION 5):
//! a 64-byte header followed by N 40-byte entries, little-endian. Membership is seqlock-
//! published by Mgmt; per-slot `active_tokens` / `isl` / `cpu_hit_blocks` are
//! AtomicU64 CAS'd by Infer Workers.

/// Magic "WKLD" (0x57 0x4B 0x4C 0x44) little-endian.
pub const MAGIC: u32 = 0x574B_4C44;
/// Layout schema version. Must match the Python reader/owner.
pub const SCHEMA_VERSION: u16 = 5;

pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 40;
pub const DEFAULT_MAX_ENTRIES: u32 = 10240;

// Header field byte offsets (see layout.py HEADER_FMT "<I H H q I I Q Q Q Q Q").
pub const OFF_MAGIC: usize = 0; // u32
pub const OFF_SCHEMA: usize = 4; // u16
pub const OFF_SEQUENCE: usize = 8; // i64 (seqlock; odd = write in progress)
pub const OFF_ENTRY_COUNT: usize = 16; // u32
pub const OFF_MAX_ENTRIES: usize = 20; // u32
pub const OFF_INSTANCE_VERSION: usize = 24; // u64
pub const OFF_HEARTBEAT: usize = 32; // u64
pub const OFF_PREFILL_SEQ: usize = 40; // u64
pub const OFF_DECODE_SEQ: usize = 48; // u64
pub const OFF_HYBRID_SEQ: usize = 56; // u64

// shm role bytes (layout.py: prefill=0, decode=1, hybrid=2, encode=3).
pub const ROLE_PREFILL: u8 = 0;
pub const ROLE_DECODE: u8 = 1;
pub const ROLE_HYBRID: u8 = 2;
pub const ROLE_ENCODE: u8 = 3;

// ---------------------------------------------------------------------------
// Schema 5: per-slot atomic CAS layout. Header is 64B; seqlock covers only membership
// (token/overlay CAS does NOT bump it), so readers must atomic-load ledger fields every pass.
// ---------------------------------------------------------------------------

// Entry field byte offsets within a 40-byte slot.
//
// The three f64 ledgers sit at 16/24/32 so that, with an 8-aligned segment base and a 40B
// stride, every slot's atomics are 8-byte aligned (mandatory on aarch64 / Ascend hosts).
pub const ENTRY_V4_OFF_INSTANCE_ID: usize = 0; // i32 (written on snapshot only)
pub const ENTRY_V4_OFF_ENDPOINT_ID: usize = 4; // i32 (written on snapshot only)
pub const ENTRY_V4_OFF_ROLE: usize = 8; // u8 (written on snapshot only)
pub const ENTRY_V4_OFF_FLAGS: usize = 9; // u8, AtomicU8 (BLOCKED / VALID)
pub const ENTRY_V4_OFF_GENERATION: usize = 10; // u16 (written on snapshot only; ABA guard)
pub const ENTRY_V4_OFF_RESERVED: usize = 12; // u32
pub const ENTRY_V4_OFF_ACTIVE_TOKENS: usize = 16; // u64 (f64::to_bits), AtomicU64
pub const ENTRY_V5_OFF_ISL: usize = 24; // u64 (f64::to_bits), AtomicU64
pub const ENTRY_V5_OFF_CPU_HIT_BLOCKS: usize = 32; // u64 (f64::to_bits), AtomicU64

// Entry flags bits.
pub const FLAG_BLOCKED: u8 = 0b0000_0001; // circuit-breaker OPEN: allocate CAS must refuse
pub const FLAG_VALID: u8 = 0b0000_0010; // slot holds a live (instance, endpoint)

const _: () = assert!(ENTRY_V5_OFF_CPU_HIT_BLOCKS + 8 <= ENTRY_SIZE);

/// Total segment size in bytes for `max_entries` slots.
pub fn total_size(max_entries: u32) -> usize {
    HEADER_SIZE + (max_entries as usize) * ENTRY_SIZE
}

/// Byte offset of the given slot's entry.
pub fn entry_offset(slot: u32) -> usize {
    HEADER_SIZE + (slot as usize) * ENTRY_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_python_layout() {
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(ENTRY_SIZE, 40);
        assert_eq!(total_size(10240), 64 + 10240 * 40);
        assert_eq!(entry_offset(0), 64);
        assert_eq!(entry_offset(1), 104);
    }

    #[test]
    fn schema5_ledgers_are_8_byte_aligned_for_every_slot() {
        for slot in 0..1024u32 {
            for field in [
                ENTRY_V4_OFF_ACTIVE_TOKENS,
                ENTRY_V5_OFF_ISL,
                ENTRY_V5_OFF_CPU_HIT_BLOCKS,
            ] {
                let off = entry_offset(slot) + field;
                assert_eq!(off % 8, 0, "slot {slot} field {field} offset {off} not 8-aligned");
            }
        }
    }
}
