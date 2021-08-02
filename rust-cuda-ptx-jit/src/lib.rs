#![deny(clippy::pedantic)]
#![cfg_attr(not(feature = "host"), no_std)]
#![feature(doc_cfg)]

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
pub mod host;

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
mod device;
