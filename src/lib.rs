#![deny(clippy::pedantic)]
#![no_std]

pub mod common;

#[cfg(feature = "host")]
pub mod host;

#[cfg(not(feature = "host"))]
pub mod device;
