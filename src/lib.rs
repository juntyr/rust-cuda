#![deny(clippy::complexity)]
#![deny(clippy::correctness)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::suspicious)]
#![allow(clippy::useless_attribute)]
#![cfg_attr(all(any(feature = "device", target_os = "cuda"), not(doc)), no_std)]
#![feature(associated_type_bounds)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(feature = "device", feature(stdsimd))]
#![cfg_attr(feature = "device", feature(asm_experimental_arch))]
#![cfg_attr(feature = "device", feature(asm_const))]
#![feature(doc_auto_cfg)]
#![feature(doc_cfg)]
#![feature(marker_trait_attr)]
#![feature(const_type_name)]
#![feature(offset_of)]
#![feature(adt_const_params)]
#![feature(impl_trait_in_assoc_type)]
#![feature(ptr_metadata)]
#![feature(decl_macro)]
#![feature(panic_info_message)]
#![feature(let_chains)]
#![feature(inline_const)]
#![feature(sync_unsafe_cell)]
#![feature(never_type)]
#![feature(layout_for_ptr)]
#![feature(cfg_version)]
#![cfg_attr(not(version("1.76.0")), feature(c_str_literals))]
#![cfg_attr(not(version("1.76.0")), feature(ptr_from_ref))]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![cfg_attr(feature = "device", feature(slice_ptr_get))]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

#[cfg(all(feature = "host", feature = "device", not(doc)))]
core::compile_error!("cannot enable the `host` and `device` features at the same time");

#[cfg(all(feature = "host", targt_os = "cuda", not(doc)))]
core::compile_error!("cannot enable the `host` feature on a target with `target_os=\"cuda\"`");

#[cfg(all(feature = "device", not(target_os = "cuda"), not(doc)))]
core::compile_error!("cannot enable the `device` feature on a target without `target_os=\"cuda\"`");

pub mod alloc;
pub mod deps;
pub mod kernel;
pub mod lend;
pub mod safety;
pub mod utils;

#[cfg(feature = "host")]
pub mod host;

#[cfg(feature = "device")]
pub mod device;
