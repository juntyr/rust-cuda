#[macro_export]
macro_rules! PtxJITConstLoad {
    ([$index:literal] => $reference:expr) => {
        unsafe {
            asm!(
                concat!("// <rust-cuda-ptx-jit-const-load-{}-", $index, "> //"),
                in(reg32) *($reference as *const _ as *const u32),
            )
        }
    };
}
