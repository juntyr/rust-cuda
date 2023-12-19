#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum CpuAndGpuKernelSignatures {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: CpuAndGpuKernelSignatures>;

#[must_use]
pub const fn check(ptx: &[u8], entry_point: &[u8]) -> CpuAndGpuKernelSignatures {
    const KERNEL_TYPE: &[u8] = b".visible .entry ";

    let mut j = 0;

    while j < ptx.len() {
        let Some(j2) = find(ptx, KERNEL_TYPE, j) else {
            return CpuAndGpuKernelSignatures::Mismatch;
        };

        if starts_with(ptx, entry_point, j2) {
            return CpuAndGpuKernelSignatures::Match;
        }

        j += 1;
    }

    CpuAndGpuKernelSignatures::Mismatch
}

const fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let mut i = 0;
    let mut j = from;

    while i < needle.len() {
        if j >= haystack.len() {
            return None;
        }

        if needle[i] == haystack[j] {
            i += 1;
            j += 1;
        } else {
            j = j + 1 - i;
            i = 0;
        }
    }

    Some(j)
}

const fn starts_with(haystack: &[u8], needle: &[u8], from: usize) -> bool {
    let mut i = 0;

    while i < needle.len() {
        if (from + i) >= haystack.len() {
            return false;
        }

        if needle[i] == haystack[from + i] {
            i += 1;
        } else {
            return false;
        }
    }

    true
}
