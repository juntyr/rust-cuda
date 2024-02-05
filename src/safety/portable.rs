macro_rules! portable_bit_semantics_docs {
    ($item:item) => {
        /// Types whose in-memory bit representation on the CPU host is safe to copy
        /// to and read back on the GPU device while maintaining the same semantics,
        /// iff the type layout on the CPU matches the type layout on the GPU.
        ///
        /// For a type to implement [`PortableBitSemantics`], it
        ///
        /// * should have the same memory layout on both the CPU and GPU, and
        ///
        /// * must not contain any references to data that are exposed as safely
        ///   accessible on both ends but actually inaccessible on one.
        ///
        /// For instance, a reference `&u8` to host memory has the same well-defined
        /// layout on both CPU and GPU (if their pointer sizes and alignments
        /// match), but it is not portable since the host memory is generally
        /// not accessible from the GPU.
        ///
        /// This trait is automatically implemented when the compiler determines
        /// it's appropriate.
        ///
        /// Note that this trait is *sealed*, i.e. you cannot implement it on your
        /// own custom types.
        ///
        /// Trait bounds usually combine [`PortableBitSemantics`] with
        /// [`TypeGraphLayout`](const_type_layout::TypeGraphLayout) to check that
        /// the type layout is indeed the same on both the host CPU and the GPU
        /// device.
        ///
        /// Types that implement [`StackOnly`](crate::safety::StackOnly) and
        /// [`TypeGraphLayout`](const_type_layout::TypeGraphLayout) satisfy both
        /// of the above criteria and thus also implement [`PortableBitSemantics`].
        $item
    };
}

#[cfg(not(doc))]
portable_bit_semantics_docs! {
    #[allow(clippy::module_name_repetitions)]
    pub trait PortableBitSemantics: sealed::PortableBitSemantics {}
}
#[cfg(doc)]
portable_bit_semantics_docs! {
    pub use sealed::PortableBitSemantics;
}

#[cfg(not(doc))]
impl<T: ?Sized + sealed::PortableBitSemantics> PortableBitSemantics for T {}

mod sealed {
    pub auto trait PortableBitSemantics {}

    impl<T: ?Sized> !PortableBitSemantics for &T {}
    impl<T: ?Sized> !PortableBitSemantics for &mut T {}
    impl<T: ?Sized> !PortableBitSemantics for *const T {}
    impl<T: ?Sized> !PortableBitSemantics for *mut T {}

    impl<T> PortableBitSemantics for core::marker::PhantomData<T> {}

    impl<T: PortableBitSemantics> PortableBitSemantics for crate::utils::ffi::DeviceConstPointer<T> {}
    impl<T: PortableBitSemantics> PortableBitSemantics for crate::utils::ffi::DeviceMutPointer<T> {}
    impl<T: PortableBitSemantics> PortableBitSemantics for crate::utils::ffi::DeviceOwnedPointer<T> {}
}
