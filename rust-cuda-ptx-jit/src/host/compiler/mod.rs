use std::ffi::{CStr, CString};

mod preprocess;
mod regex;
mod replace;

type ByteSliceOptionalArguments = Option<Box<[Option<Box<[u8]>>]>>;

#[allow(clippy::module_name_repetitions)]
pub struct PtxJITCompiler {
    ptx_slices: Box<[PtxElement]>,
    last_arguments: ByteSliceOptionalArguments,
    last_ptx: CString,
}

pub enum PtxJITResult<'s> {
    Cached(&'s CStr),
    Recomputed(&'s CStr),
}

enum PtxLoadWidth {
    B2,
    B4,
    B8,
}

enum PtxElement {
    CopiedSource {
        ptx: Box<[u8]>,
    },
    ConstLoad {
        ptx: Box<[u8]>,
        parameter_index: usize,
        byte_offset: usize,
        load_width: PtxLoadWidth,
        register: Box<[u8]>,
    },
}
