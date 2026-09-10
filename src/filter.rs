use std::hash::Hasher;

use siphasher::sip::SipHasher13;

pub struct Full;

pub enum CuckooFilter {
    Fp8(Filter<u8>),
    Fp16(Filter<u16>),
}

impl CuckooFilter {
    // constructors first, then operations: insert, contains, delete, count
}

pub trait Fingerprint: Copy + Eq {
    const EMPTY: Self; // 0 slot marker
    fn from_hash(h: u64) -> Self;
    fn to_u64(self) -> u64;
}

impl Fingerprint for u8 {
    const EMPTY: Self = 0;

    fn from_hash(h: u64) -> Self {
        let fp = (h & 0xFF) as u8;
        if fp == 0 {
            1 // since 0-value is reserved
        } else {
            fp
        }
    }

    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl Fingerprint for u16 {
    const EMPTY: Self = 0;

    fn from_hash(h: u64) -> Self {
        let fp = (h & 0xFFFF) as u16;
        if fp == 0 {
            1 // since 0-value is reserved
        } else {
            fp
        }
    }

    fn to_u64(self) -> u64 {
        self as u64
    }
}

struct Filter<F> {
    table: Vec<F>, // F = fingerprint type, u8 or u16
    bucket_count: usize,
    bucket_size: usize,
    max_kicks: u32,
    seed: [u8; 16],             // hash key for determinism
    num_items: u64,             // number of items in the filter
    victim: Option<(usize, F)>, // displaced (bucket_index, fp) parked after a failed kick chain
}

impl<F: Fingerprint> Filter<F> {
    fn hash(&self, item: &[u8]) -> u64 {
        let mut hasher = SipHasher13::new_with_key(&self.seed);
        hasher.write(item);
        hasher.finish()
    }

    fn fingerprint(&self, h: u64) -> F {
        F::from_hash(h)
    }

    fn index1(&self, h: u64) -> usize {
        // index uses bits DISJOINT from the fingerprint's low 16
        (h >> 16) as usize & (self.bucket_count - 1)
    }

    fn alt_index(&self, i: usize, fp: F) -> usize {
        // paper Eq.(2): hash the fingerprint before XOR so displaced items
        // spread across the whole table, not a 2^16 neighborhood
        let h = fp.to_u64().wrapping_mul(0x5bd1_e995); //  0x5bd1e995 is the mixing constant from MurmurHash2
        i ^ (h as usize & (self.bucket_count - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_uses_full_width() {
        // a hash with high byte set in its low 16 bits must survive
        assert_eq!(<u16 as Fingerprint>::from_hash(0xAB00), 0xAB00);
        assert_eq!(<u8 as Fingerprint>::from_hash(0xAB00), 1); // low byte 0 -> bumped
    }
}
