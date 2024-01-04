#![deny(clippy::pedantic)]
#![feature(const_type_name)]
#![feature(offset_of)]

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
