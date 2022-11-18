pub mod link;
pub mod specialise;
pub mod wrapper;

mod lints;
mod utils;

const KERNEL_TYPE_USE_START_CANARY: &str = "// <rust-cuda-kernel-param-type-use-start> //";
const KERNEL_TYPE_USE_END_CANARY: &str = "// <rust-cuda-kernel-param-type-use-end> //";
const KERNEL_TYPE_LAYOUT_IDENT: &str = "KERNEL_SIGNATURE_LAYOUT";
const PTX_CSTR_IDENT: &str = "PTX_CSTR";
