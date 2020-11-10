//! Support crate for writting GPU kernel in Rust (accel-core)
//!
//! - This crate works only for `nvptx64-nvidia-cuda` target
//! - There is no support of `libstd` for `nvptx64-nvidia-cuda` target,
//!   i.e. You need to write `#![no_std]` Rust code.
//! - `alloc` crate is supported by `PTXAllocator` which utilizes CUDA malloc/free system-calls
//!   - You can use `println!` and `assert_eq!` throught it.

extern crate alloc;

use crate::device::nvptx;
use alloc::alloc::{GlobalAlloc, Layout};

/// Memory allocator using CUDA malloc/free
pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()) as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr as *mut _);
    }
}

// Based on https://github.com/popzxc/stdext-rs/blob/master/src/macros.rs
#[macro_export]
macro_rules! function {
    () => {{
        // Okay, this is ugly, I get it. However, this is the best we can get on a stable rust.
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            core::any::type_name::<T>()
        }
        let name = type_name_of(f);
        // `3` is the length of the `::f`.
        alloc::string::String::from(&name[..name.len() - 3])
    }};
}

/// Alternative of [std::print!](https://doc.rust-lang.org/std/macro.print.html) using CUDA `vprintf` system-call
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        let msg = ::alloc::format!($($arg)*);
        #[allow(unused_unsafe)]
        unsafe {
            rust_cuda::device::nvptx::vprintf(msg.as_ptr(), ::core::ptr::null_mut());
        }
    }
}

/// Alternative of [std::println!](https://doc.rust-lang.org/std/macro.println.html) using CUDA `vprintf` system-call
#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($fmt:expr) => ($crate::print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::print!(concat!($fmt, "\n"), $($arg)*));
}

/// Assertion in GPU kernel for one expression is true.
#[macro_export]
macro_rules! assert {
    ($e:expr) => {
        if !$e {
            let msg = alloc::format!(
                "\nassertion failed: {}\nexpression: {:?}",
                stringify!($e),
                $e,
            );
            unsafe {
                rust_cuda::device::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    function!().as_ptr(),
                )
            };
        }
    };
}

/// Assertion in GPU kernel for two expressions are equal.
#[macro_export]
macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        if $a != $b {
            let msg = alloc::format!(
                "\nassertion failed: ({} == {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );
            unsafe {
                rust_cuda::device::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    function!().as_ptr(),
                )
            };
        }
    };
}

/// Assertion in GPU kernel for two expressions are not equal.
#[macro_export]
macro_rules! assert_ne {
    ($a:expr, $b:expr) => {
        if $a == $b {
            let msg = alloc::format!(
                "\nassertion failed: ({} != {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );
            unsafe {
                rust_cuda::device::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    function!().as_ptr(),
                )
            };
        }
    };
}

/// Dimension specified in kernel launching
#[derive(Debug)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Indices where the kernel code running on
#[derive(Debug)]
pub struct Idx3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[must_use]
pub fn block_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx::_block_dim_x(),
            y: nvptx::_block_dim_y(),
            z: nvptx::_block_dim_z(),
        }
    }
}

#[must_use]
pub fn block_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx::_block_idx_x(),
            y: nvptx::_block_idx_y(),
            z: nvptx::_block_idx_z(),
        }
    }
}

#[must_use]
pub fn grid_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx::_grid_dim_x(),
            y: nvptx::_grid_dim_y(),
            z: nvptx::_grid_dim_z(),
        }
    }
}

#[must_use]
pub fn thread_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx::_thread_idx_x(),
            y: nvptx::_thread_idx_y(),
            z: nvptx::_thread_idx_z(),
        }
    }
}

impl Dim3 {
    #[must_use]
    pub fn size(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Idx3 {
    #[must_use]
    pub fn as_id(&self, dim: &Dim3) -> u32 {
        self.x + self.y * dim.x + self.z * dim.x * dim.y
    }
}

#[must_use]
pub fn index() -> usize {
    let block_id = block_idx().as_id(&grid_dim());
    let thread_id = thread_idx().as_id(&block_dim());

    (block_id * block_dim().size() + thread_id) as usize
}
