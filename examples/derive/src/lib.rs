#![deny(clippy::pedantic)]
#![feature(const_type_name)]
#![feature(offset_of)]

#[derive(rust_cuda::common::LendRustToCuda)]
struct Inner<T: Copy> {
    #[cuda(embed)]
    inner: T,
}

#[derive(rust_cuda::common::LendRustToCuda)]
struct Outer<T: Copy> {
    #[cuda(embed)]
    inner: Inner<T>,
}
