#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum HostAndDeviceKernelSignatureTypeLayout {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: HostAndDeviceKernelSignatureTypeLayout>;

#[must_use]
pub const fn check(a: &[u64], b: &[u64]) -> HostAndDeviceKernelSignatureTypeLayout {
    if a.len() != b.len() {
        return HostAndDeviceKernelSignatureTypeLayout::Mismatch;
    }

    if unsafe {
        core::intrinsics::compare_bytes(
            a.as_ptr().cast(),
            b.as_ptr().cast(),
            a.len() * core::mem::size_of::<u64>(),
        ) != 0
    } {
        return HostAndDeviceKernelSignatureTypeLayout::Mismatch;
    }

    HostAndDeviceKernelSignatureTypeLayout::Match
}
