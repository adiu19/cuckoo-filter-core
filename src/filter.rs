use siphasher::sip::SipHasher13;
use std::fmt::Debug;
use std::hash::Hasher;

use crate::fingerprint::Fingerprint;

#[derive(Debug)]
pub struct Full;

#[derive(Debug, Clone)]
pub(crate) struct Filter<F> {
    pub(crate) table: Vec<F>, // F = fingerprint type, u8 or u16
    pub(crate) bucket_count: usize,
    pub(crate) bucket_size: usize,
    pub(crate) max_kicks: u32,
    pub(crate) seed: [u8; 16],             // hash key for determinism
    pub(crate) num_items: usize, // logical membership count. Maintained only by insert/delete
    pub(crate) victim: Option<(usize, F)>, // displaced (bucket_index, fp) parked after a failed kick chain
}

impl<F: Fingerprint> Filter<F> {
    pub(crate) fn with_seed(
        bucket_count: usize,
        bucket_size: usize,
        max_kicks: u32,
        seed: [u8; 16],
    ) -> Self {
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
    pub(crate) fn delete(&mut self, item: &[u8]) -> bool {
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

    pub(crate) fn insert(&mut self, item: &[u8]) -> Result<(), Full> {
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

    pub(crate) fn contains(&self, item: &[u8]) -> bool {
        let (i1, i2, fp) = self.probe(item);

        // victim check
        if self.matches_victim(i1, i2, fp) {
            return true;
        }

        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    pub(crate) fn count(&self, item: &[u8]) -> usize {
        let (i1, i2, fp) = self.probe(item);
        let mut res = usize::from(self.matches_victim(i1, i2, fp));
        res += self.bucket_count(i1, fp);
        if i1 != i2 {
            res += self.bucket_count(i2, fp);
        }
        res
    }

    pub(crate) fn census(&self) -> usize {
        self.table.iter().filter(|&&s| s != F::EMPTY).count() + usize::from(self.victim.is_some())
    }
}
