pub trait FitsIntoDeviceRegister: private::FitsIntoDeviceRegister {}
impl<T: private::FitsIntoDeviceRegister> FitsIntoDeviceRegister for T {}

mod private {
    #[marker]
    pub trait FitsIntoDeviceRegister {}
    impl<T> FitsIntoDeviceRegister for T where
        AssertTypeFitsInto64Bits<{ TypeSize::check::<T>() }>: FitsInto64Bits
    {
    }

    // Since T: Sized, the pointers are thin, and must thus fit into device
    // registers
    impl<'r, T: rustacuda_core::DeviceCopy + 'r> FitsIntoDeviceRegister
        for crate::common::DeviceConstRef<'r, T>
    {
    }
    impl<'r, T: rustacuda_core::DeviceCopy + 'r> FitsIntoDeviceRegister
        for crate::common::DeviceMutRef<'r, T>
    {
    }

    #[derive(PartialEq, Eq, core::marker::ConstParamTy)]
    pub enum TypeSize {
        TypeFitsInto64Bits,
        // FIXME: ConstParamTy variant with str ICEs in rustdoc
        #[cfg(not(doc))]
        TypeExeceeds64Bits(&'static str),
        #[cfg(doc)]
        TypeExeceeds64Bits,
    }

    impl TypeSize {
        pub const fn check<T>() -> Self {
            if core::mem::size_of::<T>() <= core::mem::size_of::<u64>() {
                Self::TypeFitsInto64Bits
            } else {
                #[cfg(not(doc))]
                {
                    Self::TypeExeceeds64Bits(core::any::type_name::<T>())
                }
                #[cfg(doc)]
                {
                    Self::TypeExeceeds64Bits
                }
            }
        }
    }

    pub enum AssertTypeFitsInto64Bits<const CHECK: TypeSize> {}

    pub trait FitsInto64Bits {}

    impl FitsInto64Bits for AssertTypeFitsInto64Bits<{ TypeSize::TypeFitsInto64Bits }> {}
}
