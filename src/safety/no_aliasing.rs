#[allow(clippy::module_name_repetitions)]
/// Types which can be safely shared between CUDA threads because they do
/// not provide safe aliasing mutable access to some shared inner state.
///
/// This trait is automatically implemented when the compiler determines
/// it's appropriate.
///
/// Data types that contain no references and can thus live entirely on
/// the stack, e.g. primitive types like [`u8`] and structs, tuples, and
/// enums made only from them, or more generally those types that implement
/// [`StackOnly`](super::StackOnly), also implement [`NoSafeAliasing`] as they
/// do not contain any inner data that might be shared when each thread is
/// given mutable access to a copy.
///
/// In contrast, `&mut T` (and any type containing a mutable reference) do *not*
/// implement [`NoSafeAliasing`] as several threads would obtain mutable
/// aliasing access to the same date, thus violating Rust's borrowing and
/// memory safety rules.
///
/// Even though `*const T` and `*mut T` do not provide *safe* mutable aliasing
/// access to their underlying data, as dereferincing them is always unsafe,
/// they (and any type containing a pointer) do *not* implement
/// [`NoSafeAliasing`] to ensure that any data type that uses them to build a
/// safe interface to accessing data, e.g. [`Box`], does not accidentially
/// implement [`NoSafeAliasing`]. If you have implemented a data structure that
/// uses `*const T` or `*mut T` internally but also ensures that no safe
/// aliasing mutable access is provided, you can *unsafely* implement
/// [`NoSafeAliasing`] for your type. Please reference the [Safety](#safety)
/// section below for more details on the contract you must uphold in this case.
///
/// # Safety
///
/// This trait must only be manually implemented for a type that upholds
/// the no-mutable-aliasing guarantee through its safe API.
///
/// The following examples outline three different cases for types that do
/// fulfil this safety requirement:
///
/// * [`Final`](final::Final) implements [`NoSafeAliasing`]
/// because even a mutable reference to it only provides read-only access
/// to its inner data.
///
/// * [`SplitSliceOverCudaThreadsConstStride`](crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride)
/// and
/// [`SplitSliceOverCudaThreadsDynamicStride`](crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride)
/// also implement [`NoSafeAliasing`] because they only provide each CUDA thread
/// with mutable access to its own partition of a slice and thus avoid mutable
/// aliasing.
///
/// * [`ThreadBlockShared`](crate::utils::shared::ThreadBlockShared)
/// and
/// [`ThreadBlockSharedSlice`](crate::utils::shared::ThreadBlockSharedSlice)
/// also implement [`NoSafeAliasing`] since they only provide access to `*mut
/// T`, which is always unsafe to mutate and thus moves the burden to uphoald
/// the no-mutable-aliasing safety invariant to the user who derefereces these
/// pointers.
pub unsafe auto trait NoSafeAliasing {}

impl<T> !NoSafeAliasing for &mut T {}
impl<T> !NoSafeAliasing for *const T {}
impl<T> !NoSafeAliasing for *mut T {}

unsafe impl<T> NoSafeAliasing for core::marker::PhantomData<T> {}

unsafe impl<T> NoSafeAliasing for r#final::Final<T> {}
unsafe impl<T: crate::common::CudaAsRust> NoSafeAliasing
    for crate::utils::aliasing::FinalCudaRepresentation<T>
{
}

unsafe impl<T, const STRIDE: usize> NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
}
unsafe impl<T> NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<T>
{
}

// Thread-block-shared data only allows unsafe aliasing since only raw pointers
//  are exposed
unsafe impl<T: 'static> NoSafeAliasing for crate::utils::shared::ThreadBlockShared<T> {}
unsafe impl<T: 'static + const_type_layout::TypeGraphLayout> NoSafeAliasing
    for crate::utils::shared::ThreadBlockSharedSlice<T>
{
}
