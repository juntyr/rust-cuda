macro_rules! stack_only_docs {
    ($item:item) => {
        /// Types which contain no pointers or references and can thus live entirely
        /// on the stack.
        ///
        /// This trait is automatically implemented when the compiler determines
        /// it's appropriate.
        ///
        /// Note that this trait is *sealed*, i.e. you cannot implement it on your
        /// own custom types.
        ///
        /// Primitive types like [`u8`] and structs, tuples, and enums made only
        /// from them implement [`StackOnly`].
        ///
        /// In contrast, `&T`, `&mut T`, `*const T`, `*mut T`, and any type
        /// containing a reference or a pointer do *not* implement [`StackOnly`].
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use rust_cuda::safety::StackOnly;
        /// fn assert_stackonly(_x: impl StackOnly) {}
        /// ```
        /// ```rust
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// assert_stackonly(42); // ok
        /// ```
        /// ```rust
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// assert_stackonly([42; 42]); // ok
        /// ```
        /// ```rust,compile_fail
        /// # use alloc::vec;
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// assert_stackonly(vec![42]); // error
        /// ```
        /// ```rust,compile_fail
        /// # use alloc::vec;
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// assert_stackonly(&42); // error
        /// ```
        /// ```rust,compile_fail
        /// # use alloc::vec;
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// # use crate::utils::shared::r#static::ThreadBlockShared;
        /// assert_stackonly(ThreadBlockShared::new_uninit()); // error
        /// ```
        /// ```rust,compile_fail
        /// # use alloc::vec;
        /// # use rust_cuda::safety::StackOnly;
        /// # fn assert_stackonly(_x: impl StackOnly) {}
        /// # use crate::utils::shared::slice::ThreadBlockSharedSlice;
        /// assert_stackonly(ThreadBlockSharedSlice::new_uninit_with_len(0)); // error
        /// ```
        $item
    };
}

#[cfg(not(doc))]
stack_only_docs! {
    #[allow(clippy::module_name_repetitions)]
    pub trait StackOnly: sealed::StackOnly {}
}
#[cfg(doc)]
stack_only_docs! {
    pub use sealed::StackOnly;
}

#[cfg(not(doc))]
impl<T: sealed::StackOnly> StackOnly for T {}

mod sealed {
    pub auto trait StackOnly {}

    impl<T: ?Sized> !StackOnly for &T {}
    impl<T: ?Sized> !StackOnly for &mut T {}
    impl<T: ?Sized> !StackOnly for *const T {}
    impl<T: ?Sized> !StackOnly for *mut T {}

    impl<T> StackOnly for core::marker::PhantomData<T> {}
}
