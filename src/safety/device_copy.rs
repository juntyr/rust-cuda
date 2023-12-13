use const_type_layout::TypeGraphLayout;

use crate::{common::DeviceAccessible, safety::StackOnly};

#[allow(clippy::module_name_repetitions)]
/// Types which are safe to memcpy from the CPU to a GPU.
///
/// For a type to implement [`SafeDeviceCopy`], it must
///
/// * have the same memory layout on both the CPU and GPU
///
/// * not contain any references to data that is inaccessible from the GPU
///
/// Types that implement both [`TypeGraphLayout`] and [`StackOnly`] satisfy
/// both of these criteria and thus implement [`SafeDeviceCopy`].
#[marker]
pub trait SafeDeviceCopy: sealed::Sealed {}

impl<T: StackOnly + TypeGraphLayout> SafeDeviceCopy for T {}
impl<T: StackOnly + TypeGraphLayout> sealed::Sealed for T {}

#[doc(hidden)]
impl<T: SafeDeviceCopy + rustacuda_core::DeviceCopy> SafeDeviceCopy for DeviceAccessible<T> {}
impl<T: SafeDeviceCopy + rustacuda_core::DeviceCopy> sealed::Sealed for DeviceAccessible<T> {}

mod sealed {
    #[marker]
    pub trait Sealed {}
}
