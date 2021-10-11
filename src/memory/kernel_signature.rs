#[derive(PartialEq, Eq)]
pub enum CpuAndGpuKernelSignatures {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: CpuAndGpuKernelSignatures>;

#[must_use]
pub const fn check(haystack: &[u8], needle: &[u8]) -> CpuAndGpuKernelSignatures {
    let mut i = 0;
    let mut j = 0;

    while i < needle.len() {
        if j >= haystack.len() {
            return CpuAndGpuKernelSignatures::Mismatch;
        }

        if needle[i] == haystack[j] {
            i += 1;
            j += 1;
        } else {
            j = j + 1 - i;
            i = 0;
        }
    }

    CpuAndGpuKernelSignatures::Match
}
