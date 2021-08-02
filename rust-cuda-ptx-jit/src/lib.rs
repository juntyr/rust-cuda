#![deny(clippy::pedantic)]
#![cfg_attr(not(feature = "host"), no_std)]

#[cfg(any(feature = "host", doc))]
pub mod host;

#[cfg(any(not(feature = "host"), doc))]
mod device;
