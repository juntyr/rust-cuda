#![deny(clippy::pedantic)]
#![feature(cfg_version)]
#![feature(const_trait_impl)]
#![feature(const_type_name)]
#![cfg_attr(not(version("1.65.0")), feature(const_ptr_offset_from))]
#![feature(const_refs_to_cell)]
#![feature(const_mut_refs)]

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
