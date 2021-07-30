#[macro_export]
macro_rules! compilePtxJITwithArguments {
    // Invocation without arguments fast track
    ($compiler:ident ()) => {
        $compiler.with_arguments(None)
    };
    // Invocation with arguments is forwarded to incremental muncher
    ($compiler:ident ( $($args:tt)* )) => {
        $crate::compilePtxJITwithArguments!(@munch None $compiler [, $($args)*] =>)
    };
    // Muncher base case: no `ConstLoad[$expr]` arguments
    (@munch None $compiler:ident [] => $($rubbish:expr),*) => {
        $compiler.with_arguments(None)
    };
    // Muncher base case: at least one `ConstLoad[$expr]` argument
    (@munch Some $compiler:ident [] => $($exprs:expr),*) => {
        $compiler.with_arguments(Some(&[$($exprs),*]))
    };
    // Muncher helper case: first `ConstLoad[$expr]` argument is recognised (redirect)
    (@munch None $compiler:ident [, ConstLoad [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler [, ConstLoad [ $head ] $($tail)*] => $($exprs),*)
    };
    // Muncher recursive case: much one `Ignore[$expr]` argument (no `ConstLoad[$expr]`s so far)
    (@munch None $compiler:ident [, Ignore [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch None $compiler [$($tail)*] => $($exprs,)* None)
    };
    // Muncher recursive case: much one `Ignore[$expr]` argument (some `ConstLoad[$expr]`s already)
    (@munch Some $compiler:ident [, Ignore [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler [$($tail)*] => $($exprs,)* None)
    };
    // Muncher recursive case: much one `ConstLoad[$expr]` (some `ConstLoad[$expr]`s already)
    (@munch Some $compiler:ident [, ConstLoad [ $head:expr ] $($tail:tt)*] => $($exprs:expr),*) => {
        $crate::compilePtxJITwithArguments!(@munch Some $compiler [$($tail)*] => $($exprs,)* Some(unsafe {
            ::std::slice::from_raw_parts($head as *const _ as *const u8, ::std::mem::size_of_val($head))
        }))
    };
}
