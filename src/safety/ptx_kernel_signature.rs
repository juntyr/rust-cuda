#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum HostAndDeviceKernelSignatureTypeLayout {
    Match,
    Mismatch,
}

pub struct Assert<const MATCH: HostAndDeviceKernelSignatureTypeLayout>;
