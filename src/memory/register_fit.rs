pub trait FitsIntoDeviceRegister: private::FitsIntoDeviceRegister {}
impl<T: private::FitsIntoDeviceRegister> FitsIntoDeviceRegister for T {}

mod private {
    pub trait FitsIntoDeviceRegister {}
    impl<T> FitsIntoDeviceRegister for T where
        AssertTypeFitsInto64Bits<{ TypeSize::check::<T>() }>: FitsInto64Bits
    {
    }

    #[derive(PartialEq, Eq)]
    pub enum TypeSize {
        TypeFitsInto64Bits,
        TypeExeceeds64Bits(&'static str),
    }

    impl TypeSize {
        pub const fn check<T>() -> Self {
            if core::mem::size_of::<T>() <= core::mem::size_of::<u64>() {
                Self::TypeFitsInto64Bits
            } else {
                Self::TypeExeceeds64Bits(core::any::type_name::<T>())
            }
        }
    }

    pub enum AssertTypeFitsInto64Bits<const CHECK: TypeSize> {}

    pub trait FitsInto64Bits {}

    impl FitsInto64Bits for AssertTypeFitsInto64Bits<{ TypeSize::TypeFitsInto64Bits }> {}
}
