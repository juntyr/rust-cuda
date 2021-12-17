#[macro_export]
#[doc(hidden)]
#[doc(cfg(not(feature = "host")))]
macro_rules! PtxJITConstLoad {
    ([$index:literal] => $reference:expr) => {
        unsafe {
            ::core::arch::asm!(
                concat!("// <rust-cuda-ptx-jit-const-load-{}-", $index, "> //"),
                in(reg32) *($reference as *const _ as *const u32),
            )
        }
    };
}
