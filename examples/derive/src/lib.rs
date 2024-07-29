#![deny(clippy::pedantic)]
#![allow(dead_code)] // FIXME: use expect
#![feature(const_type_name)]

#[derive(rc::lend::LendRustToCuda)]
#[cuda(crate = "rc")]
struct Inner<T: Copy> {
    #[cuda(embed)]
    inner: T,
}

#[derive(rc::lend::LendRustToCuda)]
#[cuda(crate = "rc")]
struct Outer<T: Copy> {
    #[cuda(embed)]
    inner: Inner<T>,
}
