#[macro_export]
#[doc(hidden)]
#[doc(cfg(not(feature = "host")))]
macro_rules! PtxJITConstLoad {
    ([$index:literal] => $reference:expr) => {
        unsafe {
            ::core::arch::asm!(
                concat!("// <rust-cuda-ptx-jit-const-load-{}-", $index, "> //"),
                in(reg32) *::core::ptr::from_ref($reference).cast::<u32>(),
            )
        }
    };
}
