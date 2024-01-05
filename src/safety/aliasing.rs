#[allow(clippy::module_name_repetitions)]
/// Types for which mutable references can be safely shared with each CUDA
/// thread without breaking Rust's no-mutable-aliasing memory safety
/// guarantees.
///
/// # Safety
///
/// A type may only implement [`NoSafeAliasing`], if and only if all of the
/// conditions below hold:
///
/// * Calling [`std::mem::replace`] on a mutable reference of the type does
///   *not* return a value which owns memory which it must deallocate on drop.
///   For instance, `&mut [T]` satisfies this criteria, but `Box<T>` does not.
///
/// * No safe alising mutable access is provided to the same memory locations
///   across multiple CUDA threads. You can use the
///   [`SplitSliceOverCudaThreadsConstStride`](crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride)
///   and
///   [`SplitSliceOverCudaThreadsDynamicStride`](crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride)
///   wrapper types to ensure that each thread is only given access to to its
///   own sub-slice partition so that aliasing is avoided.
///
/// * A mutable reference of the type must not provide mutable access to some
///   shallow inner state (in contrast to deep, which refers to values behind
///   references) of the value which the API user expects to be mutably shared
///   between all threads even if it is not in practice so as to not violate the
///   second condition. For instance, a struct `Counter { pub a: u32 }` violates
///   this third condition, as code with access to `&mut Counter` also gets
///   mutable access to its field `a` and might assume that mutations of this
///   field are either shared across threads or shared back with the host after
///   the kernel has completed, neither of which is possible. In contrast, `&mut
///   [T]` satisfies this condition, as it is well known that modifying the
///   shallow length of a slice (by assigning a sub-slice) inside a function
///   does not alter the length of the slice that the caller of the function
///   passed in.
pub unsafe trait NoSafeAliasing {}

unsafe impl<
        'a,
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const STRIDE: usize,
    > NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<&'a mut [T], STRIDE>
{
}
unsafe impl<
        'a,
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
    > NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<&'a mut [T]>
{
}

unsafe impl<
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
        const STRIDE: usize,
    > NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<
        crate::utils::exchange::buffer::CudaExchangeBuffer<T, M2D, M2H>,
        STRIDE,
    >
{
}
unsafe impl<
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
    > NoSafeAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<
        crate::utils::exchange::buffer::CudaExchangeBuffer<T, M2D, M2H>,
    >
{
}
