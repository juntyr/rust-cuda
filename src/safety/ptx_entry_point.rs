#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum HostAndDeviceKernelEntryPoint {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: HostAndDeviceKernelEntryPoint>;

#[must_use]
pub const fn check(ptx: &[u8], entry_point: &[u8]) -> HostAndDeviceKernelEntryPoint {
    const PTX_ERROR_MESSAGE: &[u8] = b"ERROR in this PTX compilation";
    const KERNEL_TYPE: &[u8] = b".visible .entry ";

    // Short-circuit to avoid extra errors when PTX compilation fails
    if ptx.len() == PTX_ERROR_MESSAGE.len() && starts_with(ptx, PTX_ERROR_MESSAGE, 0) {
        return HostAndDeviceKernelEntryPoint::Match;
    }

    let mut j = 0;

    while j < ptx.len() {
        let Some(j2) = find(ptx, KERNEL_TYPE, j) else {
            return HostAndDeviceKernelEntryPoint::Mismatch;
        };

        if starts_with(ptx, entry_point, j2) {
            return HostAndDeviceKernelEntryPoint::Match;
        }

        j += 1;
    }

    HostAndDeviceKernelEntryPoint::Mismatch
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
    let haystack_len = haystack.len() - from;
    let check_len = if needle.len() < haystack_len {
        needle.len()
    } else {
        haystack_len
    };

    let haystack = unsafe { haystack.as_ptr().add(from) };

    unsafe { core::intrinsics::compare_bytes(haystack, needle.as_ptr(), check_len) == 0 }
}
