#[allow(clippy::module_name_repetitions)]
pub trait SafeDeviceCopy: sealed::SafeDeviceCopy {}

impl<T: sealed::SafeDeviceCopy> SafeDeviceCopy for T {}

mod sealed {
    #[marker]
    pub trait SafeDeviceCopy {}

    impl<T: crate::memory::StackOnly> SafeDeviceCopy for T {}
    #[cfg(any(feature = "alloc", doc))]
    impl<T: crate::memory::UnifiedHeapOnly> SafeDeviceCopy for T {}

    impl<T: SafeDeviceCopy + rustacuda_core::DeviceCopy> SafeDeviceCopy
        for crate::common::DeviceAccessible<T>
    {
    }
    impl<T: SafeDeviceCopy> SafeDeviceCopy for crate::utils::device_copy::SafeDeviceCopyWrapper<T> {}
}
