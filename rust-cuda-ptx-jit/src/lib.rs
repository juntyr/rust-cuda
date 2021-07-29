#![deny(clippy::pedantic)]
#![cfg_attr(not(feature = "host"), no_std)]

#[cfg(feature = "host")]
pub mod host;

#[cfg(not(feature = "host"))]
mod device;
