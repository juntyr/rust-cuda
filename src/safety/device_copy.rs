#[allow(clippy::module_name_repetitions)]
pub trait SafeDeviceCopy: sealed::SafeDeviceCopy {}

impl<T: sealed::SafeDeviceCopy> SafeDeviceCopy for T {}

mod sealed {
    #[marker]
    pub trait SafeDeviceCopy {}

    // Thread-block-shared data cannot be copied since information is added inside
    //  CUDA
    impl<T: 'static> !SafeDeviceCopy for crate::utils::shared::r#static::ThreadBlockShared<T> {}
    impl<T: 'static + const_type_layout::TypeGraphLayout> !SafeDeviceCopy
        for crate::utils::shared::slice::ThreadBlockSharedSlice<T>
    {
    }

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
}
