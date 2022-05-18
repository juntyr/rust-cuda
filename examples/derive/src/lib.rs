#![deny(clippy::pedantic)]
#![feature(const_trait_impl)]
#![feature(const_type_name)]
#![feature(const_ptr_offset_from)]
#![feature(const_refs_to_cell)]
#![feature(const_mut_refs)]

#[derive(rust_cuda::common::LendRustToCuda)]
struct Inner<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[r2cEmbed]
    inner: T,
}

#[derive(rust_cuda::common::LendRustToCuda)]
struct Outer<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[r2cEmbed]
    inner: Inner<T>,
}
