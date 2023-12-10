//! Check that the CPU and NVIDIA GPU architecture are sufficiently
//!  aligned such that FFI-safe types have the same bit representation
//!  on both sides.
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static BOOL: bool; }
//! ```
//!
//! ```compile_fail
//! #[deny(improper_ctypes)]
//! extern "C" { static CHAR: char; }
//! ```
//!
//! ```compile_fail
//! #[deny(improper_ctypes)]
//! extern "C" { static STR: &str; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static F32: f32; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static F64: f64; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static I8: i8; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static I16: i16; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static I32: i32; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static I64: i64; }
//! ```
//!
//! ```compile_fail
//! #[deny(improper_ctypes)]
//! extern "C" { static I128: i128; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static ISIZE: isize; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static U8: u8; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static U16: u16; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static U32: u32; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static U64: u64; }
//! ```
//!
//! ```compile_fail
//! #[deny(improper_ctypes)]
//! extern "C" { static U128: u128; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static USIZE: usize; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static CONST_PTR: *const u8; }
//! ```
//!
//! ```
//! #[deny(improper_ctypes)]
//! extern "C" { static MUT_PTR: *mut u8; }
//! ```

#[cfg(not(target_endian = "little"))]
compile_error!("Both the CPU and GPU must use little-endian representation.");

#[cfg(not(target_pointer_width = "64"))]
compile_error!("Both the CPU and GPU must use 64bit pointers.");

#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
struct TypeLayout {
    size: usize,
    align: usize,
}

#[allow(dead_code)]
struct Assert<const LAYOUT: TypeLayout>;

macro_rules! size_align_check {
    ($ty:ty[$size:literal : $align:literal]) => {
        const _: Assert<
            {
                TypeLayout {
                    size: $size,
                    align: $align,
                }
            },
        > = Assert::<
            {
                TypeLayout {
                    size: core::mem::size_of::<$ty>(),
                    align: core::mem::align_of::<$ty>(),
                }
            },
        >;
    };
}

size_align_check! { bool [1:1] }
size_align_check! { f32 [4:4] }
size_align_check! { f64 [8:8] }
size_align_check! { i8 [1:1] }
size_align_check! { i16 [2:2] }
size_align_check! { i32 [4:4] }
size_align_check! { i64 [8:8] }
size_align_check! { isize [8:8] }
size_align_check! { u8 [1:1] }
size_align_check! { u16 [2:2] }
size_align_check! { u32 [4:4] }
size_align_check! { u64 [8:8] }
size_align_check! { usize [8:8] }

size_align_check! { (bool, bool) [2:1] }
size_align_check! { (f32, f32) [8:4] }
size_align_check! { (f64, f64) [16:8] }
size_align_check! { (i8, i8) [2:1] }
size_align_check! { (i16, i16) [4:2] }
size_align_check! { (i32, i32) [8:4] }
size_align_check! { (i64, i64) [16:8] }
size_align_check! { (isize, isize) [16:8] }
size_align_check! { (u8, u8) [2:1] }
size_align_check! { (u16, u16) [4:2] }
size_align_check! { (u32, u32) [8:4] }
size_align_check! { (u64, u64) [16:8] }
size_align_check! { (usize, usize) [16:8] }
