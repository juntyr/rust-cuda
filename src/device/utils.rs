/// Abort the CUDA kernel using the `trap` system call.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn abort() -> ! {
    unsafe { ::core::arch::nvptx::trap() }
}

/// The [`print`](print()) function takes an [`Arguments`](core::fmt::Arguments)
/// struct and formats and prints it to the CUDA kernel's standard output using
/// the `vprintf` system call.
///
/// The [`Arguments`](core::fmt::Arguments) instance can be created with the
/// [`format_args!`](core::format_args) macro.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn print(args: ::core::fmt::Arguments) {
    let msg; // place to store the dynamically expanded format string
    let msg = if let Some(msg) = args.as_str() {
        msg
    } else {
        msg = ::alloc::fmt::format(args);
        msg.as_str()
    };

    unsafe {
        ::core::arch::nvptx::vprintf(msg.as_ptr(), ::core::ptr::null_mut());
    }
}
