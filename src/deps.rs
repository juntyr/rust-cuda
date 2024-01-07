#[doc(hidden)]
pub extern crate alloc;

pub extern crate const_type_layout;

#[cfg(feature = "host")]
pub extern crate owning_ref;

#[cfg(feature = "host")]
pub extern crate rustacuda;

pub extern crate rustacuda_core;
