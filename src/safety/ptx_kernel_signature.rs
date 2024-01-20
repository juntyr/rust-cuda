const SIGNATURE_ERROR_MESSAGE: &[u8] = b"ERROR in this PTX compilation";

#[marker]
pub trait SameHostAndDeviceKernelSignatureTypeLayout<const A: &'static [u8], const B: &'static [u8]>
{
}

impl<const AB: &'static [u8]> SameHostAndDeviceKernelSignatureTypeLayout<AB, AB> for () {}
impl<const A: &'static [u8]> SameHostAndDeviceKernelSignatureTypeLayout<A, SIGNATURE_ERROR_MESSAGE>
    for ()
{
}
impl<const B: &'static [u8]> SameHostAndDeviceKernelSignatureTypeLayout<SIGNATURE_ERROR_MESSAGE, B>
    for ()
{
}

pub const fn check<const A: &'static [u8], const B: &'static [u8]>()
where
    (): SameHostAndDeviceKernelSignatureTypeLayout<A, B>,
{
}
