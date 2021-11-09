use const_type_layout::{serialise_type_graph, serialised_type_graph_len, TypeGraph};

#[derive(PartialEq, Eq)]
pub enum CpuAndGpuTypeLayouts {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: CpuAndGpuTypeLayouts>;

#[must_use]
pub const fn check<T: ~const TypeGraph>(device: &'static [u8]) -> CpuAndGpuTypeLayouts
where
    [u8; serialised_type_graph_len::<T>()]:,
{
    let host = serialise_type_graph::<T>();

    if host.len() != device.len() {
        return CpuAndGpuTypeLayouts::Mismatch;
    }

    let mut i = 0;

    while i < host.len() {
        if host[i] != device[i] {
            return CpuAndGpuTypeLayouts::Mismatch;
        }

        i += 1;
    }

    CpuAndGpuTypeLayouts::Match
}
