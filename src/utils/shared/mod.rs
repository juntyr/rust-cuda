pub mod slice;
pub mod r#static;

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
#[allow(clippy::module_name_repetitions)]
pub trait ThreadBlockShared: 'static + Sized {
    fn share_uninit() -> r#static::ThreadBlockShared<Self>;
}

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
impl<T: 'static> ThreadBlockShared for T {
    fn share_uninit() -> r#static::ThreadBlockShared<Self> {
        r#static::ThreadBlockShared::uninit()
    }
}

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
pub trait ThreadBlockSharedSlice: 'static {
    type Elem: Sized;

    fn share_uninit(len: usize) -> slice::ThreadBlockSharedSlice<Self::Elem>;
}

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
impl<T: 'static> ThreadBlockSharedSlice for [T] {
    type Elem = T;

    fn share_uninit(len: usize) -> slice::ThreadBlockSharedSlice<T> {
        slice::ThreadBlockSharedSlice::with_len(len)
    }
}
