//! # ptx-builder &emsp; [![CI Status]][workflow] [![MSRV]][repo] [![Rust Doc]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/rust-ptx-builder/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/rust-ptx-builder/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.80.0--nightly-orange
//! [repo]: https://github.com/juntyr/rust-ptx-builder
//!
//! [Rust Doc]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/rust-ptx-builder/

/// Error handling.
#[macro_use]
pub mod error;

/// External executables that are needed to build CUDA crates.
pub mod executable;

/// Build helpers.
pub mod builder;

/// Build reporting helpers.
pub mod reporter;

mod source;

/// Convenient re-exports of mostly used types.
pub mod prelude {
    pub use crate::{
        builder::{BuildStatus, Builder, CrateType, MessageFormat, Profile},
        reporter::{CargoAdapter, ErrorLogPrinter},
    };
}
