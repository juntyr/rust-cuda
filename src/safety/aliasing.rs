#[expect(clippy::module_name_repetitions)]
/// Types for which mutable references can be safely shared with each CUDA
/// thread without breaking Rust's no-mutable-aliasing memory safety
/// guarantees.
///
/// # Safety
///
/// A type may only implement [`SafeMutableAliasing`], if and
/// only if all of the safety conditions below hold:
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
///   second condition. For instance, `Vec<T>` violates this third condition, as
///   code with access to `&mut Vec<T>` can also mutate the length of the
///   vector, which is shallow state that is expected to be propagated to the
///   caller of a function sharing this vector (it is also related to the deep
///   contents of the vector via a safety invariant) and might thus assume that
///   mutations of this length are either shared across threads or shared back
///   with the host after the kernel has completed, neither of which is
///   possible. In contrast, `&mut [T]` satisfies this condition, as it is well
///   known that modifying the shallow length of a slice (by assigning a
///   sub-slice) inside a function does not alter the length of the slice that
///   the caller of the function passed in.
pub unsafe trait SafeMutableAliasing {}

unsafe impl<
        'a,
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const STRIDE: usize,
    > SafeMutableAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<&'a mut [T], STRIDE>
{
}

unsafe impl<
        'a,
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
    > SafeMutableAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<&'a mut [T]>
{
}

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
        const STRIDE: usize,
    > SafeMutableAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<
        crate::utils::exchange::buffer::CudaExchangeBuffer<T, M2D, M2H>,
        STRIDE,
    >
{
}

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<
        T: crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + const_type_layout::TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
    > SafeMutableAliasing
    for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<
        crate::utils::exchange::buffer::CudaExchangeBuffer<T, M2D, M2H>,
    >
{
}
