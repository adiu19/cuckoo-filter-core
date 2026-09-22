use std::fmt::Debug;

pub(crate) trait Fingerprint: Copy + Eq + Debug {
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
