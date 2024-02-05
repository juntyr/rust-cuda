use const_type_layout::{serialise_type_graph, serialised_type_graph_len, TypeGraphLayout};

#[allow(clippy::module_name_repetitions)]
#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum HostAndDeviceKernelSignatureTypeLayout {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: HostAndDeviceKernelSignatureTypeLayout>;

#[must_use]
pub const fn check<T: TypeGraphLayout>(
    device: &'static [u8],
) -> HostAndDeviceKernelSignatureTypeLayout
where
    [u8; serialised_type_graph_len::<T>()]:,
{
    const SIGNATURE_ERROR_MESSAGE: &[u8] = b"ERROR in this PTX compilation";

    // Short-circuit to avoid extra errors when PTX compilation fails
    if equals(device, SIGNATURE_ERROR_MESSAGE) {
        return HostAndDeviceKernelSignatureTypeLayout::Match;
    }

    let host = serialise_type_graph::<T>();

    if equals(device, &host) {
        HostAndDeviceKernelSignatureTypeLayout::Match
    } else {
        HostAndDeviceKernelSignatureTypeLayout::Mismatch
    }
}

const fn equals(device: &[u8], host: &[u8]) -> bool {
    if device.len() != host.len() {
        return false;
    }

    unsafe { core::intrinsics::compare_bytes(device.as_ptr(), host.as_ptr(), device.len()) == 0 }
}
