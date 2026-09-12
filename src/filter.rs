use siphasher::sip::SipHasher13;
use std::fmt::Debug;
use std::hash::Hasher;

#[derive(Debug)]
pub struct Full;

#[derive(PartialEq, Debug)]
pub enum BuildError {
    ZeroCapacity,
    ZeroBucketSize,
}

#[derive(Debug)]
pub struct CuckooFilter(Inner);

#[derive(Debug)]
enum Inner {
    Fp8(Filter<u8>),
    Fp16(Filter<u16>),
}

impl CuckooFilter {
    pub fn with_seed_fp8(
        capacity: usize,
        bucket_size: usize,
        max_kicks: u32,
        seed: [u8; 16],
    ) -> Result<Self, BuildError> {
        if capacity == 0 {
            return Err(BuildError::ZeroCapacity);
        }

        if bucket_size == 0 {
            return Err(BuildError::ZeroBucketSize);
        }

        Ok(CuckooFilter(Inner::Fp8(Filter::<u8>::with_seed(
            capacity.div_ceil(bucket_size).next_power_of_two(),
            bucket_size,
            max_kicks,
            seed,
        ))))
    }

    pub fn with_seed_fp16(
        capacity: usize,
        bucket_size: usize,
        max_kicks: u32,
        seed: [u8; 16],
    ) -> Result<Self, BuildError> {
        if capacity == 0 {
            return Err(BuildError::ZeroCapacity);
        }

        if bucket_size == 0 {
            return Err(BuildError::ZeroBucketSize);
        }

        Ok(CuckooFilter(Inner::Fp16(Filter::<u16>::with_seed(
            capacity.div_ceil(bucket_size).next_power_of_two(),
            bucket_size,
            max_kicks,
            seed,
        ))))
    }

    pub fn insert(&mut self, item: &[u8]) -> Result<(), Full> {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.insert(item),
            CuckooFilter(Inner::Fp16(f)) => f.insert(item),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.num_items,
            CuckooFilter(Inner::Fp16(f)) => f.num_items,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, item: &[u8]) -> bool {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.contains(item),
            CuckooFilter(Inner::Fp16(f)) => f.contains(item),
        }
    }
}

trait Fingerprint: Copy + Eq + Debug {
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
        u64::from(self)
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
        u64::from(self)
    }
}

#[derive(Debug)]
struct Filter<F> {
    table: Vec<F>, // F = fingerprint type, u8 or u16
    bucket_count: usize,
    bucket_size: usize,
    max_kicks: u32,
    seed: [u8; 16],             // hash key for determinism
    num_items: usize,           // number of items in the filter
    victim: Option<(usize, F)>, // displaced (bucket_index, fp) parked after a failed kick chain
}

impl<F: Fingerprint> Filter<F> {
    fn with_seed(bucket_count: usize, bucket_size: usize, max_kicks: u32, seed: [u8; 16]) -> Self {
        Filter {
            table: vec![F::EMPTY; bucket_count * bucket_size],
            bucket_count,
            bucket_size,
            max_kicks,
            seed,
            num_items: 0,
            victim: None,
        }
    }
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

    fn bucket_contains(&self, bucket_idx: usize, fp: F) -> bool {
        let start = bucket_idx * self.bucket_size;
        self.table[start..start + self.bucket_size].contains(&fp)
    }

    fn bucket_insert(&mut self, bucket_idx: usize, fp: F) -> bool {
        let start = bucket_idx * self.bucket_size;
        for i in start..start + self.bucket_size {
            if self.table[i] == F::EMPTY {
                self.table[i] = fp;
                return true;
            }
        }

        false
    }

    fn bucket_displace(&mut self, bucket_idx: usize, fp: F, kick_id: u32) -> F {
        let start = bucket_idx * self.bucket_size;
        let candidate = start + (fp.to_u64() as usize + kick_id as usize) % self.bucket_size;

        let orphan = self.table[candidate];
        self.table[candidate] = fp;
        orphan
    }

    fn initiate_kick(&mut self, bid: usize, fp: F) -> Result<(), Full> {
        if self.victim.is_some() {
            return Err(Full);
        }

        self.num_items += 1; // we are going to accept the insert

        // init state
        let mut num_kicks = 0;
        let mut orphan = fp;
        let mut target_bid = bid;
        while num_kicks < self.max_kicks {
            if self.bucket_insert(target_bid, orphan) {
                return Ok(());
            }

            orphan = self.bucket_displace(target_bid, orphan, num_kicks);
            target_bid = self.alt_index(target_bid, orphan);
            num_kicks += 1;
        }

        self.victim = Some((target_bid, orphan));
        Ok(())
    }

    fn insert(&mut self, item: &[u8]) -> Result<(), Full> {
        let h = self.hash(item);
        let fp = self.fingerprint(h);

        let i1 = self.index1(h);
        let i2 = self.alt_index(i1, fp);

        // try i1, then i2
        if self.bucket_insert(i1, fp) || self.bucket_insert(i2, fp) {
            self.num_items += 1;
            return Ok(());
        }

        // start kick chain; always i1 for now
        // the insert on i1 is retried, but that's fine and a noop since it is going to fail and start displacements
        self.initiate_kick(i1, fp)
    }

    fn contains(&self, item: &[u8]) -> bool {
        let h = self.hash(item);
        let fp = self.fingerprint(h);
        let i1 = self.index1(h);
        let i2 = self.alt_index(i1, fp);
        // victim check
        if let Some((vb, vfp)) = self.victim {
            if vfp == fp && (vb == i1 || vb == i2) {
                return true;
            }
        }

        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u8; 16] = [7u8; 16];

    #[test]
    fn u8_build_failure() {
        assert_eq!(
            BuildError::ZeroCapacity,
            CuckooFilter::with_seed_fp8(0, 1, 2, SEED).err().unwrap()
        );

        assert_eq!(
            BuildError::ZeroBucketSize,
            CuckooFilter::with_seed_fp8(2, 0, 2, SEED).err().unwrap()
        );
    }

    #[test]
    fn u16_build_failure() {
        assert_eq!(
            BuildError::ZeroCapacity,
            CuckooFilter::with_seed_fp16(0, 1, 2, SEED).err().unwrap()
        );

        assert_eq!(
            BuildError::ZeroBucketSize,
            CuckooFilter::with_seed_fp16(2, 0, 2, SEED).err().unwrap()
        );
    }

    #[test]
    fn u16_uses_full_width() {
        // a hash with high byte set in its low 16 bits must survive
        assert_eq!(<u16 as Fingerprint>::from_hash(0xAB00), 0xAB00);
        assert_eq!(<u8 as Fingerprint>::from_hash(0xAB00), 1); // low byte 0 -> bumped
    }

    #[test]
    fn u8_insert_bucketsize_2() {
        let mut cf8 = CuckooFilter::with_seed_fp8(8, 2, 2, SEED).unwrap();

        // fill upto 75%
        for item in 1u64..7u64 {
            assert!(cf8.insert(&item.to_le_bytes()).is_ok());
        }

        assert_eq!(6, cf8.len());

        for item in 1u64..7u64 {
            assert!(cf8.contains(&item.to_le_bytes()));
        }
    }

    #[test]
    fn u8_insert_bucketsize_4() {
        let mut cf8 = CuckooFilter::with_seed_fp8(8, 4, 2, SEED).unwrap();

        // fill upto 75%
        for item in 1u64..7u64 {
            assert!(cf8.insert(&item.to_le_bytes()).is_ok());
        }

        assert_eq!(6, cf8.len());

        for item in 1u64..7u64 {
            assert!(cf8.contains(&item.to_le_bytes()));
        }
    }

    #[test]
    fn identical_sequences_produce_identical_tables() {
        let mut cf = CuckooFilter::with_seed_fp8(256, 4, 100, SEED).unwrap();
        let mut cf_alt = CuckooFilter::with_seed_fp8(256, 4, 100, SEED).unwrap();

        let mut failed = 0;
        for item in 0u64..300 {
            let ra = cf.insert(&item.to_le_bytes());
            let rb = cf_alt.insert(&item.to_le_bytes());

            assert_eq!(ra.is_ok(), rb.is_ok(), "diverged at item {item}");

            if ra.is_err() {
                failed += 1;
            }
        }

        assert!(failed > 0, "expected saturation");

        let (CuckooFilter(Inner::Fp8(fa)), CuckooFilter(Inner::Fp8(fb))) = (&cf, &cf_alt) else {
            unreachable!()
        };

        assert_eq!(fa.table, fb.table);
        assert_eq!(fa.victim, fb.victim);
        assert_eq!(fa.num_items, fb.num_items);
    }
}
