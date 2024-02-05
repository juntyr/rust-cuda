use crate::deps::alloc::{fmt, string::String};

/// Abort the CUDA kernel using the `trap` system call.
///
/// [`abort`] poisons the CUDA context and no more work can be performed in it.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn abort() -> ! {
    unsafe { ::core::arch::nvptx::trap() }
}

/// Exit the CUDA kernel using the `exit` instruction.
///
/// # Safety
///
/// [`exit`] quits the kernel early and any mutable data accessible outside this
/// kernel launch (by the host or a subsequent kernel launch) may be in an
/// inconsistent state. Therefore, kernel failure must be communicated back to
/// host and handled in some other manner.
///
/// Safely return from the main kernel function instead.
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn exit() -> ! {
    unsafe { ::core::arch::asm!("exit;", options(noreturn)) }
}

/// Prints to the CUDA kernel's standard output using the `vprintf` system call.
///
/// Replacement for the [`std::print!`] macro, which now forwards to the
/// [`print()`] function.
pub macro print($($arg:tt)*) {
    self::print(::core::format_args!($($arg)*))
}

/// Prints to the CUDA kernel's standard output using the `vprintf` system call.
///
/// Replacement for the [`std::println!`] macro, which now forwards to the
/// [`print()`] function.
pub macro println {
    () => {
        self::print(::core::format_args!("\n"))
    },
    ($($arg:tt)*) => {
        self::print(::core::format_args!("{}\n", ::core::format_args!($($arg)*)))
    },
}

/// The [`print()`] function takes an [`Arguments`](core::fmt::Arguments) struct
/// and formats and prints it to the CUDA kernel's standard output using the
/// `vprintf` system call.
///
/// The [`Arguments`](core::fmt::Arguments) instance can be created with the
/// [`format_args!`](core::format_args) macro.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn print(args: ::core::fmt::Arguments) {
    #[repr(C)]
    struct FormatArgs {
        msg_len: u32,
        msg_ptr: *const u8,
    }

    let msg; // place to store the dynamically expanded format string
    #[allow(clippy::option_if_let_else)]
    let msg = if let Some(msg) = args.as_str() {
        msg
    } else {
        msg = fmt::format(args);
        msg.as_str()
    };

    let args = FormatArgs {
        msg_len: u32::try_from(msg.len()).unwrap_or(u32::MAX),
        msg_ptr: msg.as_ptr(),
    };

    unsafe {
        ::core::arch::nvptx::vprintf(c"%*s".as_ptr().cast(), ::core::ptr::from_ref(&args).cast());
    }
}

/// Helper function to efficiently pretty-print a [`core::panic::PanicInfo`]
/// using the `vprintf` system call.
///
/// If `allow_dynamic_message` is set,
/// [`alloc::fmt::format`](crate::deps::alloc::fmt::format) is used to print
/// [`core::panic::PanicInfo::message`] message when
/// [`core::fmt::Arguments::as_str`] returns [`None`]. Note that this may pull
/// in a large amount of string formatting and dynamic allocation code.
/// If unset, a default placeholder panic message is printed instead.
///
/// If `allow_dynamic_payload` is set, [`core::panic::PanicInfo::payload`] is
/// checked for [`&str`] and [`String`] to get a message to print if
/// [`core::panic::PanicInfo::message`] returns [`None`]. Note that this may
/// pull in some dynamic dispatch code. If unset, a default placeholder panic
/// message is printed instead.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn pretty_print_panic_info(
    info: &::core::panic::PanicInfo,
    allow_dynamic_message: bool,
    allow_dynamic_payload: bool,
) {
    #[repr(C)]
    struct FormatArgs {
        file_len: u32,
        file_ptr: *const u8,
        line: u32,
        column: u32,
        thread_idx_x: u32,
        thread_idx_y: u32,
        thread_idx_z: u32,
        msg_len: u32,
        msg_ptr: *const u8,
    }

    let msg; // place to store the dynamically expanded format string
    #[allow(clippy::option_if_let_else)]
    let msg = if let Some(message) = info.message() {
        if let Some(msg) = message.as_str() {
            msg
        } else if allow_dynamic_message {
            msg = fmt::format(*message);
            msg.as_str()
        } else {
            "<dynamic panic message>"
        }
    } else if let Some(msg) = info.payload().downcast_ref::<&'static str>()
        && allow_dynamic_payload
    {
        msg
    } else if let Some(msg) = info.payload().downcast_ref::<String>()
        && allow_dynamic_payload
    {
        msg.as_str()
    } else {
        "<unknown panic payload type>"
    };

    let location_line = info.location().map_or(0, ::core::panic::Location::line);
    let location_column = info.location().map_or(0, ::core::panic::Location::column);
    let location_file = info
        .location()
        .map_or("<unknown panic location>", ::core::panic::Location::file);

    let thread_idx = crate::device::thread::Thread::this().idx();

    let args = FormatArgs {
        file_len: u32::try_from(location_file.len()).unwrap_or(u32::MAX),
        file_ptr: location_file.as_ptr(),
        line: location_line,
        column: location_column,
        thread_idx_x: thread_idx.x,
        thread_idx_y: thread_idx.y,
        thread_idx_z: thread_idx.z,
        msg_len: u32::try_from(msg.len()).unwrap_or(u32::MAX),
        msg_ptr: msg.as_ptr(),
    };

    unsafe {
        ::core::arch::nvptx::vprintf(
            c"panicked at %*s:%u:%u on thread (x=%u, y=%u, z=%u):\n%*s\n"
                .as_ptr()
                .cast(),
            ::core::ptr::from_ref(&args).cast(),
        );
    }
}

/// Helper function to efficiently pretty-print an error message (inside an
/// allocation error handler) using the `vprintf` system call.
#[track_caller]
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn pretty_print_alloc_error(layout: ::core::alloc::Layout) {
    #[repr(C)]
    struct FormatArgs {
        size: usize,
        align: usize,
        file_len: u32,
        file_ptr: *const u8,
        line: u32,
        column: u32,
        thread_idx_x: u32,
        thread_idx_y: u32,
        thread_idx_z: u32,
    }

    let location = ::core::panic::Location::caller();
    let thread_idx = crate::device::thread::Thread::this().idx();

    let args = FormatArgs {
        size: layout.size(),
        align: layout.align(),
        file_len: u32::try_from(location.file().len()).unwrap_or(u32::MAX),
        file_ptr: location.file().as_ptr(),
        line: location.line(),
        column: location.column(),
        thread_idx_x: thread_idx.x,
        thread_idx_y: thread_idx.y,
        thread_idx_z: thread_idx.z,
    };

    unsafe {
        ::core::arch::nvptx::vprintf(
            c"memory allocation of %llu bytes with alignment %llu failed at \
            %*s:%u:%u on thread (x=%u, y=%u, z=%u)\n"
                .as_ptr()
                .cast(),
            ::core::ptr::from_ref(&args).cast(),
        );
    }
}
