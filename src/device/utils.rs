use alloc::alloc::{GlobalAlloc, Layout};
use core::arch::nvptx;

/// Memory allocator using CUDA malloc/free
pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()).cast()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr.cast());
    }
}

// Based on https://github.com/popzxc/stdext-rs/blob/master/src/macros.rs
#[macro_export]
#[doc(hidden)]
macro_rules! function {
    () => {{
        // Hack to get the name of the enclosing function
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            core::any::type_name::<T>()
        }
        let name = type_name_of(f);

        // Remove the `::f` suffix
        &name[..name.len() - 3]
    }};
}

/// Alternative of [`std::print!`](https://doc.rust-lang.org/std/macro.print.html) using CUDA `vprintf` system-call
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        let msg = $crate::alloc::format!($($arg)*);

        #[allow(unused_unsafe)]
        unsafe {
            ::core::arch::nvptx::vprintf(msg.as_ptr(), ::core::ptr::null_mut());
        }
    }
}

/// Alternative of [`std::println!`](https://doc.rust-lang.org/std/macro.println.html) using CUDA `vprintf` system-call
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($fmt:expr) => ($crate::print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::print!(concat!($fmt, "\n"), $($arg)*));
}

/// Assertion in GPU kernel for one expression is true.
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
#[macro_export]
macro_rules! assert {
    ($e:expr) => {
        if !$e {
            let msg = $crate::alloc::format!(
                "\nassertion failed: {}\nexpression: {:?}",
                stringify!($e),
                $e,
            );

            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    $crate::function!().as_ptr(),
                )
            };
        }
    };
}

/// Assertion in GPU kernel for two expressions are equal.
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
#[macro_export]
macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        if $a != $b {
            let msg = $crate::alloc::format!(
                "\nassertion failed: ({} == {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );

            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    $crate::function!().as_ptr(),
                )
            };
        }
    };
}

/// Assertion in GPU kernel for two expressions are not equal.
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
#[macro_export]
macro_rules! assert_ne {
    ($a:expr, $b:expr) => {
        if $a == $b {
            let msg = $crate::alloc::format!(
                "\nassertion failed: ({} != {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );

            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    $crate::function!().as_ptr(),
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

/// Indices that the kernel code is running on
#[derive(Debug)]
pub struct Idx3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[must_use]
pub fn block_dim() -> Dim3 {
    #[allow(clippy::cast_sign_loss)]
    unsafe {
        Dim3 {
            x: nvptx::_block_dim_x() as u32,
            y: nvptx::_block_dim_y() as u32,
            z: nvptx::_block_dim_z() as u32,
        }
    }
}

#[must_use]
pub fn block_idx() -> Idx3 {
    #[allow(clippy::cast_sign_loss)]
    unsafe {
        Idx3 {
            x: nvptx::_block_idx_x() as u32,
            y: nvptx::_block_idx_y() as u32,
            z: nvptx::_block_idx_z() as u32,
        }
    }
}

#[must_use]
pub fn grid_dim() -> Dim3 {
    #[allow(clippy::cast_sign_loss)]
    unsafe {
        Dim3 {
            x: nvptx::_grid_dim_x() as u32,
            y: nvptx::_grid_dim_y() as u32,
            z: nvptx::_grid_dim_z() as u32,
        }
    }
}

#[must_use]
pub fn thread_idx() -> Idx3 {
    #[allow(clippy::cast_sign_loss)]
    unsafe {
        Idx3 {
            x: nvptx::_thread_idx_x() as u32,
            y: nvptx::_thread_idx_y() as u32,
            z: nvptx::_thread_idx_z() as u32,
        }
    }
}

impl Dim3 {
    #[must_use]
    pub fn size(&self) -> usize {
        (self.x as usize) * (self.y as usize) * (self.z as usize)
    }
}

impl Idx3 {
    #[must_use]
    pub fn as_id(&self, dim: &Dim3) -> usize {
        (self.x as usize)
            + (self.y as usize) * (dim.x as usize)
            + (self.z as usize) * (dim.x as usize) * (dim.y as usize)
    }
}

#[must_use]
pub fn index() -> usize {
    let block_id = block_idx().as_id(&grid_dim());
    let thread_id = thread_idx().as_id(&block_dim());

    block_id * block_dim().size() + thread_id
}
