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

#[derive(Debug, Clone)]
pub struct CuckooFilter(Inner);

#[derive(Debug, Clone)]
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

    pub fn delete(&mut self, item: &[u8]) -> bool {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.delete(item),
            CuckooFilter(Inner::Fp16(f)) => f.delete(item),
        }
    }

    #[must_use]
    pub fn count(&self, item: &[u8]) -> usize {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.count(item),
            CuckooFilter(Inner::Fp16(f)) => f.count(item),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.num_items,
            CuckooFilter(Inner::Fp16(f)) => f.num_items,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
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

#[derive(Debug, Clone)]
struct Filter<F> {
    table: Vec<F>, // F = fingerprint type, u8 or u16
    bucket_count: usize,
    bucket_size: usize,
    max_kicks: u32,
    seed: [u8; 16],             // hash key for determinism
    num_items: usize,           // logical membership count. Maintained only by insert/delete
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

    fn probe(&self, item: &[u8]) -> (usize, usize, F) {
        let h = self.hash(item);
        let fp = self.fingerprint(h);

        let i1 = self.index1(h);
        let i2 = self.alt_index(i1, fp);

        (i1, i2, fp)
    }

    fn bucket_range(&self, i: usize) -> std::ops::Range<usize> {
        let start = i * self.bucket_size;
        start..start + self.bucket_size
    }

    fn index1(&self, h: u64) -> usize {
        // index uses bits DISJOINT from the fingerprint's low 16
        ((h >> 16) & (self.bucket_count as u64 - 1)) as usize
    }

    fn alt_index(&self, i: usize, fp: F) -> usize {
        // paper Eq.(2): hash the fingerprint before XOR so displaced items
        // spread across the whole table, not a 2^16 neighborhood
        let h = fp.to_u64().wrapping_mul(0x5bd1_e995); //  0x5bd1e995 is the mixing constant from MurmurHash2
        i ^ ((h & (self.bucket_count as u64 - 1)) as usize)
    }

    fn bucket_contains(&self, bucket_idx: usize, fp: F) -> bool {
        self.table[self.bucket_range(bucket_idx)].contains(&fp)
    }

    fn bucket_count(&self, bucket_idx: usize, fp: F) -> usize {
        self.table[self.bucket_range(bucket_idx)]
            .iter()
            .filter(|&&f| f == fp)
            .count()
    }

    fn bucket_insert(&mut self, bucket_idx: usize, fp: F) -> bool {
        for i in self.bucket_range(bucket_idx) {
            if self.table[i] == F::EMPTY {
                self.table[i] = fp;
                return true;
            }
        }

        false
    }

    fn bucket_remove(&mut self, bucket_idx: usize, fp: F) -> bool {
        for i in self.bucket_range(bucket_idx) {
            if self.table[i] == fp {
                self.table[i] = F::EMPTY;
                return true;
            }
        }

        false
    }

    fn bucket_displace(&mut self, bucket_idx: usize, fp: F, kick_id: u32) -> F {
        let range = self.bucket_range(bucket_idx);
        let candidate = range.start + (fp.to_u64() as usize + kick_id as usize) % self.bucket_size;

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

    fn try_rehome_victim(&mut self) {
        if let Some((vb, vfp)) = self.victim {
            if self.bucket_insert(vb, vfp) || self.bucket_insert(self.alt_index(vb, vfp), vfp) {
                self.victim = None;
            }
        }
    }

    fn matches_victim(&self, i1: usize, i2: usize, fp: F) -> bool {
        if let Some((vb, vfp)) = self.victim {
            if vfp == fp && (i1 == vb || i2 == vb) {
                return true;
            }
        }

        false
    }

    /// Removes one copy of item fingerprint, checking the victim slot, then
    /// bucket i1, then i2 (fixed order, determinism requires it). Returns whether
    /// a matching fingerprint was found and removed.
    ///
    /// After a successful table removal, a parked victim is re-homed into its own
    /// two buckets if space opened up.
    ///
    /// Deleting an item that was never inserted may remove a colliding item's
    /// entry and introduce a false negative: only delete items known to have been
    /// inserted.
    fn delete(&mut self, item: &[u8]) -> bool {
        let (i1, i2, fp) = self.probe(item);

        // victim has what we need
        if self.matches_victim(i1, i2, fp) {
            self.num_items -= 1;
            self.victim = None;
            return true;
        }

        // one of i1 and i2 have what we need
        if self.bucket_remove(i1, fp) || self.bucket_remove(i2, fp) {
            self.num_items -= 1;
            self.try_rehome_victim();
            return true;
        }

        false
    }

    fn insert(&mut self, item: &[u8]) -> Result<(), Full> {
        let (i1, i2, fp) = self.probe(item);

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
        let (i1, i2, fp) = self.probe(item);

        // victim check
        if self.matches_victim(i1, i2, fp) {
            return true;
        }

        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    fn count(&self, item: &[u8]) -> usize {
        let (i1, i2, fp) = self.probe(item);
        let mut res = usize::from(self.matches_victim(i1, i2, fp));
        res += self.bucket_count(i1, fp);
        if i1 != i2 {
            res += self.bucket_count(i2, fp);
        }
        res
    }

    #[cfg(test)]
    fn census(&self) -> usize {
        self.table.iter().filter(|&&s| s != F::EMPTY).count() + usize::from(self.victim.is_some())
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
    fn u8_insert_b2() {
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
    fn u8_insert_b4() {
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
    fn u8_delete_b4() {
        let mut cf8 = CuckooFilter::with_seed_fp8(8, 4, 2, SEED).unwrap();

        // fill upto 75%
        for item in 1u64..7u64 {
            assert!(cf8.insert(&item.to_le_bytes()).is_ok());
        }

        assert_eq!(6, cf8.len());

        for item in 1u64..3u64 {
            assert!(cf8.delete(&item.to_le_bytes()));
        }

        assert_eq!(4, cf8.len());

        for item in 3u64..7u64 {
            assert!(cf8.contains(&item.to_le_bytes()));
        }
    }

    #[test]
    fn u8_count_b4() {
        let mut cf8 = CuckooFilter::with_seed_fp8(8, 4, 2, SEED).unwrap();

        // insert 1u64 4 times
        let item = 1u64.to_le_bytes();
        for _ in 0..4 {
            assert!(cf8.insert(&item).is_ok());
        }

        assert_eq!(4, cf8.len());
        assert_eq!(4, cf8.count(&item));

        for _ in 0..2 {
            assert!(cf8.delete(&item));
        }

        assert_eq!(2, cf8.len());
        assert_eq!(2, cf8.count(&item));

        // 2u64 is never inserted, so it's count should be 0
        assert_eq!(0, cf8.count(&2u64.to_le_bytes()));
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

        assert_eq!(fa.num_items, fa.census());
        assert_eq!(fb.num_items, fb.census());

        assert_eq!(fa.table, fb.table);
        assert_eq!(fa.victim, fb.victim);
        assert_eq!(fa.num_items, fb.num_items);

        for item in 0u64..50 {
            let ra = cf.delete(&item.to_le_bytes());
            let rb = cf_alt.delete(&item.to_le_bytes());

            assert_eq!(ra, rb, "deletion diverged at item {item}");
        }

        let (CuckooFilter(Inner::Fp8(fa)), CuckooFilter(Inner::Fp8(fb))) = (&cf, &cf_alt) else {
            unreachable!()
        };

        assert_eq!(fa.num_items, fa.census());
        assert_eq!(fb.num_items, fb.census());
        assert_eq!(fa.table, fb.table);
        assert_eq!(fa.victim, fb.victim);
        assert_eq!(fa.num_items, fb.num_items);
    }
}
