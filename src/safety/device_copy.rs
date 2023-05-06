#[allow(clippy::module_name_repetitions)]
pub trait SafeDeviceCopy: sealed::SafeDeviceCopy {}

impl<T: sealed::SafeDeviceCopy> SafeDeviceCopy for T {}

mod sealed {
    #[marker]
    pub trait SafeDeviceCopy {}

    impl<T: crate::safety::StackOnly> SafeDeviceCopy for T {}
    #[cfg(any(feature = "alloc", doc))]
    impl<T: crate::safety::UnifiedHeapOnly> SafeDeviceCopy for T {}

    impl<T: SafeDeviceCopy + rustacuda_core::DeviceCopy> SafeDeviceCopy
        for crate::common::DeviceAccessible<T>
    {
    }
    impl<T: SafeDeviceCopy + const_type_layout::TypeGraphLayout> SafeDeviceCopy
        for crate::utils::device_copy::SafeDeviceCopyWrapper<T>
    {
    }

    // No data is actually copied to the device
    impl<T: 'static> SafeDeviceCopy for crate::utils::shared::r#static::ThreadBlockShared<T> {}
}
