use crate::filter::Filter;
use crate::fingerprint::Fingerprint;

const MAGIC: [u8; 2] = *b"CF";
const HEADER_LEN: usize = 2 + 1 + 1 + 8 + 8 + 8 + 4 + 16 + 1 + 8;

const MAX_BUCKET_COUNT: u64 = 1 << 32;
const MAX_BUCKET_SIZE: u64 = 255;

const FORMAT_VERSION: u8 = 1; // filter cannot carry old versions at runtime

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
    pub(crate) fn export(&self) -> Vec<u8> {
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

    pub(crate) fn import(bytes: &[u8]) -> Result<Self, ImportError> {
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
