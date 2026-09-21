use siphasher::sip::SipHasher13;
use std::fmt::Debug;
use std::hash::Hasher;

const MAGIC: [u8; 2] = *b"CF";
const HEADER_LEN: usize = 2 + 1 + 1 + 8 + 8 + 8 + 4 + 16 + 1 + 8;

const MAX_BUCKET_COUNT: u64 = 1 << 32;
const MAX_BUCKET_SIZE: u64 = 255;

const FORMAT_VERSION: u8 = 1; // filter cannot carry old versions at runtime

#[derive(Debug)]
pub struct Full;

#[derive(Debug)]
pub enum ImportError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u8),
    UnsupportedFpBits(u8),
    FpBitsMismatch { expected: u8, got: u8 },
    InvalidFingerprint,
    InvalidVictimFlag(u8),
    SizeMismatch,
    InvalidBucketCount,
    InvalidBucketSize,
    BucketCountTooLarge,
    BucketSizeTooLarge,
    InvalidNumItems,
    InvalidVictimBucketId,
    InvalidVictimFp,
    CensusMismatch,
}

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
    /// Serializes the filter into the versioned byte format; see `Filter::export`
    /// for the full layout.
    pub fn export(&self) -> Vec<u8> {
        match self {
            CuckooFilter(Inner::Fp8(f)) => f.export(),
            CuckooFilter(Inner::Fp16(f)) => f.export(),
        }
    }

    pub fn import(bytes: &[u8]) -> Result<Self, ImportError> {
        if bytes.len() < 4 {
            return Err(ImportError::TooShort);
        }

        match bytes[3] {
            8 => Ok(CuckooFilter(Inner::Fp8(Filter::<u8>::import(bytes)?))),
            16 => Ok(CuckooFilter(Inner::Fp16(Filter::<u16>::import(bytes)?))),
            v => Err(ImportError::UnsupportedFpBits(v)),
        }
    }

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
    const BITS: u8;
    fn from_hash(h: u64) -> Self;
    fn to_u64(self) -> u64;
    fn write_le(self, buf: &mut Vec<u8>);
    fn read_le(bytes: &[u8], pos: &mut usize) -> Self;
}

impl Fingerprint for u8 {
    const EMPTY: Self = 0;
    const BITS: u8 = 8;
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

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8], pos: &mut usize) -> Self {
        let res = u8::from_le_bytes(bytes[*pos..*pos + 1].try_into().unwrap());
        *pos += 1;
        res
    }
}

impl Fingerprint for u16 {
    const EMPTY: Self = 0;
    const BITS: u8 = 16;

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

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8], pos: &mut usize) -> Self {
        let res = u16::from_le_bytes(bytes[*pos..*pos + 2].try_into().unwrap());
        *pos += 2;
        res
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
    /// Serializes the filter into the version-1 byte format.
    ///
    /// All multi-byte integers are little-endian.
    ///
    /// | bytes | field | validation on import |
    /// |---|---|---|
    /// | 2 | magic `CF` | must equal `MAGIC` |
    /// | 1 | format version | must be a known version |
    /// | 1 | fingerprint bits | must be 8 or 16 |
    /// | 8 | num_items (u64) | ≤ slots + 1; must equal recount of table + victim |
    /// | 8 | bucket_count (u64) | nonzero, power of two, below sanity cap |
    /// | 8 | bucket_size (u64) | nonzero, below sanity cap |
    /// | 4 | max_kicks (u32) | — |
    /// | 16 | seed | — |
    /// | 1 | victim flag | must be 0 or 1 |
    /// | 8 | victim bucket id (u64) | if flag=1: < bucket_count; if flag=0: must be 0 |
    /// | fp_bits/8 | victim fingerprint | if flag=1: nonzero; if flag=0: must be 0 |
    /// | bucket_count × bucket_size × fp_bits/8 | table slots, bucket-major | total input length must match exactly |
    ///
    /// The version also covers how hashing works (hash function, bit split,
    /// alt-index constant) — changing any of those needs a version bump.
    ///
    /// `export` writes the newest version; `import` reads all shipped versions
    /// and errors on unknown ones.
    fn export(&self) -> Vec<u8> {
        let header = HEADER_LEN + (F::BITS as usize / 8);
        let mut buf = Vec::with_capacity(header + self.table.len() * (F::BITS as usize / 8));

        buf.extend(&MAGIC);
        buf.push(FORMAT_VERSION);
        buf.extend_from_slice(&(F::BITS).to_le_bytes());
        buf.extend_from_slice(&((self.num_items as u64).to_le_bytes()));
        buf.extend_from_slice(&((self.bucket_count as u64).to_le_bytes()));
        buf.extend_from_slice(&((self.bucket_size as u64).to_le_bytes()));
        buf.extend_from_slice(&((self.max_kicks).to_le_bytes()));
        buf.extend_from_slice(&self.seed);
        buf.push(u8::from(self.victim.is_some()));

        // serialize victim
        match self.victim {
            Some((vb, vfp)) => {
                buf.extend_from_slice(&((vb as u64).to_le_bytes()));
                vfp.write_le(&mut buf);
            }
            None => {
                buf.extend_from_slice(&((0u64).to_le_bytes()));
                F::EMPTY.write_le(&mut buf);
            }
        }

        for &item in &self.table {
            item.write_le(&mut buf);
        }

        buf
    }

    fn import(bytes: &[u8]) -> Result<Self, ImportError> {
        let header = HEADER_LEN + (F::BITS as usize / 8);

        if bytes.len() < header {
            return Err(ImportError::TooShort);
        }

        let mut pos: usize = 0;

        let magic = read_bytes(bytes, 2, &mut pos);
        if magic != MAGIC {
            return Err(ImportError::BadMagic);
        }

        let version = read_bytes(bytes, 1, &mut pos)[0];
        if version != FORMAT_VERSION {
            return Err(ImportError::UnsupportedVersion(version));
        }

        let fp_bits = read_bytes(bytes, 1, &mut pos)[0];
        if fp_bits != F::BITS {
            return Err(ImportError::FpBitsMismatch {
                expected: F::BITS,
                got: fp_bits,
            });
        }
        let num_items =
            usize::try_from(read_64(bytes, &mut pos)).map_err(|_| ImportError::InvalidNumItems)?;

        let bucket_count = read_64(bytes, &mut pos);
        if !bucket_count.is_power_of_two() {
            return Err(ImportError::InvalidBucketCount);
        }

        if bucket_count > MAX_BUCKET_COUNT {
            return Err(ImportError::BucketCountTooLarge);
        }

        let bucket_size = read_64(bytes, &mut pos);
        if bucket_size == 0 {
            return Err(ImportError::InvalidBucketSize);
        }

        if bucket_size > MAX_BUCKET_SIZE {
            return Err(ImportError::BucketSizeTooLarge);
        }

        let slots = usize::try_from(bucket_count * bucket_size)
            .map_err(|_| ImportError::BucketCountTooLarge)?;

        if bytes.len() != header + slots * (F::BITS as usize / 8) {
            return Err(ImportError::SizeMismatch);
        }

        if num_items > slots + 1 {
            return Err(ImportError::InvalidNumItems);
        }

        let max_kicks = read_32(bytes, &mut pos);
        let seed: [u8; 16] = read_bytes(bytes, 16, &mut pos).try_into().unwrap();
        let is_victim_present: bool = match read_bytes(bytes, 1, &mut pos)[0] {
            0 => false,
            1 => true,
            v => return Err(ImportError::InvalidVictimFlag(v)),
        };

        let vb = read_64(bytes, &mut pos);
        let vfp = F::read_le(bytes, &mut pos);

        if is_victim_present {
            if vb >= bucket_count {
                return Err(ImportError::InvalidVictimBucketId);
            }

            if vfp == F::EMPTY {
                return Err(ImportError::InvalidVictimFp);
            }
        } else {
            if vb != 0 {
                return Err(ImportError::InvalidVictimBucketId);
            }

            if vfp != F::EMPTY {
                return Err(ImportError::InvalidVictimFp);
            }
        }

        let mut table = Vec::with_capacity(slots);
        for _ in 0..slots {
            table.push(F::read_le(bytes, &mut pos));
        }

        let filter = Filter {
            table,
            bucket_count: bucket_count as usize,
            bucket_size: bucket_size as usize,
            max_kicks,
            seed,
            num_items,
            victim: is_victim_present.then_some((vb as usize, vfp)),
        };

        if filter.num_items != filter.census() {
            return Err(ImportError::CensusMismatch);
        }

        Ok(filter)
    }

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

    fn census(&self) -> usize {
        self.table.iter().filter(|&&s| s != F::EMPTY).count() + usize::from(self.victim.is_some())
    }
}

fn read_64(bytes: &[u8], pos: &mut usize) -> u64 {
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn read_32(bytes: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    v
}

fn read_bytes<'a>(bytes: &'a [u8], n: usize, pos: &mut usize) -> &'a [u8] {
    let s = &bytes[*pos..*pos + n];
    *pos += n;
    s
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

    #[test]
    fn export_import_roundtrip() {
        let mut cf = CuckooFilter::with_seed_fp8(256, 4, 100, SEED).unwrap();

        let mut failed = 0;
        for item in 0u64..300 {
            let ra = cf.insert(&item.to_le_bytes());
            if ra.is_err() {
                failed += 1;
            }
        }
        assert!(failed > 0, "expected saturation");

        let bytes = cf.export();
        let cf_alt = CuckooFilter::import(&bytes).unwrap();

        assert_eq!(cf_alt.export(), bytes);

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
