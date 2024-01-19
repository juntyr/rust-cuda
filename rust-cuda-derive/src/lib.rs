//! [![CI Status]][workflow] [![MSRV]][repo] [![Rust Doc]][docs] [![License
//! Status]][fossa] [![Code Coverage]][codecov] [![Gitpod
//! Ready-to-Code]][gitpod]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/rust-cuda/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/rust-cuda/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.77.0--nightly-orange
//! [repo]: https://github.com/juntyr/rust-cuda
//!
//! [Rust Doc]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/rust-cuda/rust_cuda_derive/
//!
//! [License Status]: https://app.fossa.com/api/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda.svg?type=shield
//! [fossa]: https://app.fossa.com/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda?ref=badge_shield
//!
//! [Code Coverage]: https://img.shields.io/codecov/c/github/juntyr/rust-cuda?token=wfeAeybbbx
//! [codecov]: https://codecov.io/gh/juntyr/rust-cuda
//!
//! [Gitpod Ready-to-Code]: https://img.shields.io/badge/Gitpod-ready-blue?logo=gitpod
//! [gitpod]: https://gitpod.io/#https://github.com/juntyr/rust-cuda
//!
//! `rust-cuda-derive` provides the [`#[derive(LendRustToCuda)`](LendRustToCuda)
//! derive macro for the
//! [`rust_cuda::lend::RustToCuda`]
//! utility trait, which enables the usage of the
//! [`rust_cuda::lend::LendToCuda`]
//! trait that allows Rust data structures to be shared with CUDA kernels.
//!
//! The async variants of both traits are *optionally* implemented as well.
//!
//! [`rust_cuda::lend::RustToCuda`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.RustToCuda.html
//! [`rust_cuda::lend::LendToCuda`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.LendToCuda.html

#![deny(clippy::complexity)]
#![deny(clippy::correctness)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::suspicious)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error;

use proc_macro::TokenStream;

mod rust_to_cuda;

#[proc_macro_error]
#[proc_macro_derive(LendRustToCuda, attributes(cuda))]
/// Provides the [`#[derive(LendRustToCuda)`](LendRustToCuda)
/// derive macro for the
/// [`rust_cuda::lend::RustToCuda`]
/// utility trait, which enables the usage of the
/// [`rust_cuda::lend::LendToCuda`]
/// trait that allows Rust data structures to be shared with CUDA kernels.
///
/// At the moment, only
/// [`struct`](https://doc.rust-lang.org/std/keyword.struct.html)s are supported
/// by this derive macro.
///
/// The derive also accepts a `#[cuda(...)]` attribute. You can annotate the
/// entire struct with the `#[cuda(...)]` to configure the implementation as
/// follows:
///
/// - `#[cuda(crate = "<crate-path>")]` changes the path to the [`rust-cuda`]
///   crate that the derive uses, which by default is `rust_cuda`.
/// - `#[cuda(bound = "<where-predicate>")]` adds the provided predicate to the
///   where clause of the trait implementation.
/// - `#[cuda(free = "<type>")]` removes the the auto-added trait bounds for the
///   type parameter `<type>` from the trait implementation, e.g. when
///   implementing a wrapper around [`std::marker::PhantomData<T>`] which should
///   implement the trait for any `T`.
/// - `#[cuda(async = <bool>)]` explicitly enables or disables the async
///   implementation of the trait, [`rust_cuda::lend::RustToCudaAsync`]. By
///   default, `#[cuda(async = true)]` is set.
/// - `#[cuda(layout::ATTR = "VALUE")]` adds the `#[layout(ATTR = "VALUE")]`
///   attribute to the [`#derive(const_type_layout::TypeLayout)`] derive for
///   this struct's [`rust_cuda::lend::RustToCuda::CudaRepresentation`].
/// - `#[cuda(ignore)]` removes all subsequent attributes from the generated
///   [`rust_cuda::lend::RustToCuda::CudaRepresentation`] struct.
///
/// Additionally, the `#[cuda(...)]` attribute can also be applied individually
/// to the fields of the struct to customise the implementation as follows:
///
/// - `#[cuda(embed)]` signals that this field has a non-identity CUDA
///   representation and should be embedded by using the
///   [`rust_cuda::lend::RustToCuda`] implementation of this field's type. When
///   this attribute is not specified, the field must instead implement
///   [`Copy`], [`rust_cuda::safety::PortableBitSemantics`], and
///   [`const_type_layout::TypeGraphLayout`].
/// - `#[cuda(embed = "<proxy-type>")]` works like `#[cuda(embed)]` but can be
///   used when the field's type does not implement
///   [`rust_cuda::lend::RustToCuda`] itself, but some `<proxy-type>` exists,
///   which implements [`rust_cuda::lend::RustToCudaProxy`] for the field's
///   type.
/// - `#[cuda(ignore)]` removes all subsequent attributes from this field in the
///   generated [`rust_cuda::lend::RustToCuda::CudaRepresentation`] struct.
///
/// [`rust_cuda::lend::RustToCuda`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.RustToCuda.html
/// [`rust_cuda::lend::LendToCuda`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.LendToCuda.html
/// [`rust-cuda`]: https://juntyr.github.io/rust-cuda/rust_cuda
/// [`rust_cuda::lend::RustToCudaAsync`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.RustToCudaAsync.html
/// [`#derive(const_type_layout::TypeLayout)`]: https://docs.rs/const-type-layout/0.2.1/const_type_layout/derive.TypeLayout.html
/// [`rust_cuda::lend::RustToCuda::CudaRepresentation`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.RustToCuda.html#associatedtype.CudaRepresentation
/// [`rust_cuda::safety::PortableBitSemantics`]: https://juntyr.github.io/rust-cuda/rust_cuda/safety/trait.PortableBitSemantics.html
/// [`const_type_layout::TypeGraphLayout`]: https://docs.rs/const-type-layout/0.2.1/const_type_layout/trait.TypeGraphLayout.html
/// [`rust_cuda::lend::RustToCudaProxy`]: https://juntyr.github.io/rust-cuda/rust_cuda/lend/trait.RustToCudaProxy.html
pub fn rust_to_cuda_derive(input: TokenStream) -> TokenStream {
    // Note: We cannot report a more precise span yet
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `RustToCuda` and `CudaAsRust` traits
    rust_to_cuda::impl_rust_to_cuda(&ast)
}
