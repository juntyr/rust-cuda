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
