#[macro_export]
#[doc(hidden)]
#[doc(cfg(feature = "host"))]
#[allow(clippy::module_name_repetitions)]
macro_rules! compilePtxJITwithArguments {
    // Invocation without arguments fast track
    ($compiler:ident ()) => {
        $crate::compilePtxJITwithArguments!($compiler.with_arguments ())
    };
    // Invocation without arguments fast track
    ($compiler:ident $(. $path:ident)+ ()) => {
        $compiler$(.$path)+(None)
    };
    // Invocation with arguments is forwarded to incremental muncher
    ($compiler:ident ( $($args:tt)* )) => {
        $crate::compilePtxJITwithArguments!($compiler.with_arguments ( $($args)* ))
    };
    // Invocation with arguments is forwarded to incremental muncher
    ($compiler:ident $(. $path:ident)+ ( $($args:tt)* )) => {
        $crate::compilePtxJITwithArguments!(@munch None $compiler$(.$path)+ => [, $($args)*] =>)
    };
    // Muncher base case: no `ConstLoad[$expr]` arguments
    (@munch None $compiler:ident $(. $path:ident)+ => [] => $($rubbish:expr),*) => {
        $compiler$(.$path)+(None)
    };
    // Muncher base case: at least one `ConstLoad[$expr]` argument
    (@munch Some $compiler:ident $(. $path:ident)+ => [] => $($exprs:expr),*) => {
        $compiler$(.$path)+(Some(&[$($exprs),*]))
    };
    // Muncher helper case: first `ConstLoad[$expr]` argument is recognised (redirect)
    (@munch None $compiler:ident $(. $path:ident)+ => [, ConstLoad [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler$(.$path)+ => [, ConstLoad [ $head ] $($tail)*] => $($exprs),*)
    };
    // Muncher recursive case: much one `Ignore[$expr]` argument (no `ConstLoad[$expr]`s so far)
    (@munch None $compiler:ident $(. $path:ident)+ => [, Ignore [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch None $compiler$(.$path)+ => [$($tail)*] => $($exprs,)* None)
    };
    // Muncher recursive case: much one `Ignore[$expr]` argument (some `ConstLoad[$expr]`s already)
    (@munch Some $compiler:ident $(. $path:ident)+ => [, Ignore [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler$(.$path)+ => [$($tail)*] => $($exprs,)* None)
    };
    // Muncher recursive case: much one `ConstLoad[$expr]` (some `ConstLoad[$expr]`s already)
    (@munch Some $compiler:ident $(. $path:ident)+ => [, ConstLoad [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler$(.$path)+ => [$($tail)*] => $($exprs,)* Some(unsafe {
            ::std::slice::from_raw_parts(::std::ptr::from_ref($head).cast::<u8>(), ::std::mem::size_of_val($head))
        }))
    };
}
