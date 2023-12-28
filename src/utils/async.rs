#[cfg(feature = "host")]
use std::{
    future::Future, future::IntoFuture, future::Ready, marker::PhantomData, sync::Arc, sync::Mutex,
    task::Poll, task::Waker,
};

#[cfg(feature = "host")]
use rustacuda::{
    error::CudaError, error::CudaResult, event::Event, event::EventFlags, event::EventStatus,
    stream::Stream, stream::StreamWaitEventFlags,
};

#[cfg(feature = "host")]
use crate::host::CudaDropWrapper;

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub trait CudaAsync<'stream, T>: Sized + IntoFuture<Output = CudaResult<T>> {
    /// Wraps a still-asynchronous `value` which is being computed on `stream`
    /// such that its computation can be synchronised on.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    fn new(value: T, stream: &'stream Stream) -> CudaResult<Self>;

    /// Synchronises on this computation to block until it has completed and
    /// the inner value can be safely returned and again be used in synchronous
    /// operations.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    fn synchronize(self) -> CudaResult<T>;

    /// Moves the asynchronous data move to a different [`Stream`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    fn move_to_stream<'stream_new>(
        self,
        stream: &'stream_new Stream,
    ) -> CudaResult<impl CudaAsync<'stream_new, T>>;
}

#[cfg(feature = "host")]
pub struct Sync<T> {
    value: T,
}

#[cfg(feature = "host")]
impl<'stream, T> CudaAsync<'stream, T> for Sync<T> {
    fn new(value: T, _stream: &'stream Stream) -> CudaResult<Self> {
        Ok(Self { value })
    }

    fn synchronize(self) -> CudaResult<T> {
        Ok(self.value)
    }

    #[allow(refining_impl_trait)]
    fn move_to_stream(self, _stream: &Stream) -> CudaResult<Self> {
        Ok(self)
    }
}

#[cfg(feature = "host")]
impl<T> IntoFuture for Sync<T> {
    type IntoFuture = Ready<CudaResult<T>>;
    type Output = CudaResult<T>;

    fn into_future(self) -> Self::IntoFuture {
        std::future::ready(Ok(self.value))
    }
}

#[cfg(feature = "host")]
pub struct Async<'stream, T> {
    _stream: PhantomData<&'stream Stream>,
    event: CudaDropWrapper<Event>,
    waker: Arc<Mutex<Option<Waker>>>,
    value: T,
}

#[cfg(feature = "host")]
impl<'stream, T> CudaAsync<'stream, T> for Async<'stream, T> {
    fn new(value: T, stream: &'stream Stream) -> CudaResult<Self> {
        let event = CudaDropWrapper::from(Event::new(
            EventFlags::DISABLE_TIMING | EventFlags::BLOCKING_SYNC,
        )?);
        event.record(stream)?;

        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let waker_callback = waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut waker) = waker_callback.lock() {
                if let Some(waker) = waker.take() {
                    waker.wake();
                }
            }
        }))?;

        Ok(Self {
            _stream: PhantomData::<&'stream Stream>,
            event,
            waker,
            value,
        })
    }

    fn synchronize(self) -> CudaResult<T> {
        self.event.synchronize()?;

        Ok(self.value)
    }

    #[allow(refining_impl_trait)]
    fn move_to_stream<'stream_new>(
        self,
        stream: &'stream_new Stream,
    ) -> CudaResult<Async<'stream_new, T>> {
        stream.wait_event(&self.event, StreamWaitEventFlags::DEFAULT)?;
        self.event.record(stream)?;

        let waker_callback = self.waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut waker) = waker_callback.lock() {
                if let Some(waker) = waker.take() {
                    waker.wake();
                }
            }
        }))?;

        Ok(Async {
            _stream: PhantomData::<&'stream_new Stream>,
            event: self.event,
            waker: self.waker,
            value: self.value,
        })
    }
}

#[cfg(feature = "host")]
impl<'stream, T> Async<'stream, T> {
    /// # Safety
    ///
    /// The returned inner value of type `T` may not yet have completed its
    /// asynchronous work and may thus be in an inconsistent state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub unsafe fn unwrap_unchecked(self) -> T {
        self.value
    }
}

#[cfg(feature = "host")]
impl<'stream, T> IntoFuture for Async<'stream, T> {
    type Output = CudaResult<T>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let mut wrapper = Some(self);

        std::future::poll_fn(move |cx| match &wrapper {
            Some(Async { waker, event, .. }) => match event.query() {
                Ok(EventStatus::NotReady) => waker.lock().map_or_else(
                    |_| Poll::Ready(Err(CudaError::OperatingSystemError)),
                    |mut waker| {
                        *waker = Some(cx.waker().clone());
                        Poll::Pending
                    },
                ),
                Ok(EventStatus::Ready) => match wrapper.take() {
                    Some(Async { value, .. }) => Poll::Ready(Ok(value)),
                    None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
                },
                Err(err) => Poll::Ready(Err(err)),
            },
            None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
        })
    }
}
