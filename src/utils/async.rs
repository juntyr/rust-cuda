#[cfg(feature = "host")]
use std::{
    future::Future, future::IntoFuture, future::Ready, marker::PhantomData, sync::Arc, sync::Mutex,
    task::Poll, task::Waker,
};

#[cfg(feature = "host")]
use rustacuda::{
    error::CudaError, error::CudaResult, event::Event, event::EventFlags, stream::Stream,
    stream::StreamWaitEventFlags,
};

#[cfg(feature = "host")]
use crate::host::CudaDropWrapper;

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub trait CudaAsync<'stream, T, C: Send = ()>: Sized + IntoFuture<Output = CudaResult<T>> {
    /// Wraps a still-asynchronous `value` which is being computed on `stream`
    /// such that its computation can be synchronised on.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    fn new(
        value: T,
        stream: &'stream Stream,
        capture: C,
        on_completion: impl Send + FnOnce(C) -> CudaResult<()>,
    ) -> CudaResult<Self>;

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
    ) -> CudaResult<impl CudaAsync<'stream_new, T, C>>;
}

#[cfg(feature = "host")]
pub struct Sync<T> {
    value: T,
}

#[cfg(feature = "host")]
impl<'stream, T, C: Send> CudaAsync<'stream, T, C> for Sync<T> {
    fn new(
        value: T,
        _stream: &'stream Stream,
        capture: C,
        on_completion: impl Send + FnOnce(C) -> CudaResult<()>,
    ) -> CudaResult<Self> {
        on_completion(capture)?;

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
pub struct Async<'stream, T, C = ()> {
    _stream: PhantomData<&'stream Stream>,
    event: CudaDropWrapper<Event>,
    value: T,
    status: Arc<Mutex<AsyncStatus<C>>>,
}

// This could also be expressed as a
//  https://docs.rs/oneshot/latest/oneshot/index.html channel
#[cfg(feature = "host")]
enum AsyncStatus<C> {
    Processing { waker: Option<Waker>, capture: C },
    Completed { result: CudaResult<()> },
}

// TODO: completion is NOT allowed to make any cuda calls
#[cfg(feature = "host")]
impl<'stream, T, C: Send> CudaAsync<'stream, T, C> for Async<'stream, T, C> {
    fn new(
        value: T,
        stream: &'stream Stream,
        capture: C,
        on_completion: impl Send + FnOnce(C) -> CudaResult<()>,
    ) -> CudaResult<Self> {
        let event = CudaDropWrapper::from(Event::new(
            EventFlags::DISABLE_TIMING | EventFlags::BLOCKING_SYNC,
        )?);

        let status = Arc::new(Mutex::new(AsyncStatus::Processing {
            waker: None,
            capture,
        }));

        let status_callback = status.clone();
        stream.add_callback(Box::new(move |res| {
            let Ok(mut status) = status_callback.lock() else {
                return;
            };

            let old_status =
                std::mem::replace(&mut *status, AsyncStatus::Completed { result: Ok(()) });

            let AsyncStatus::Processing { mut waker, capture } = old_status else {
                // this path should never be taken
                *status = old_status;
                return;
            };

            if let Err(err) = res {
                *status = AsyncStatus::Completed { result: Err(err) };
            } else if let Err(err) = on_completion(capture) {
                *status = AsyncStatus::Completed { result: Err(err) };
            }

            if let Some(waker) = waker.take() {
                waker.wake();
            }
        }))?;

        event.record(stream)?;

        Ok(Self {
            _stream: PhantomData::<&'stream Stream>,
            event,
            value,
            status,
        })
    }

    fn synchronize(self) -> CudaResult<T> {
        let Ok(status) = self.status.lock() else {
            return Err(CudaError::OperatingSystemError);
        };

        if let AsyncStatus::Completed { result } = &*status {
            return result.map(|()| self.value);
        }

        std::mem::drop(status);

        self.event.synchronize()?;

        let Ok(status) = self.status.lock() else {
            return Err(CudaError::OperatingSystemError);
        };

        match &*status {
            AsyncStatus::Completed { result } => result.map(|()| self.value),
            AsyncStatus::Processing { .. } => Err(CudaError::NotReady),
        }
    }

    #[allow(refining_impl_trait)]
    fn move_to_stream<'stream_new>(
        self,
        stream: &'stream_new Stream,
    ) -> CudaResult<Async<'stream_new, T, C>> {
        let Ok(status) = self.status.lock() else {
            return Err(CudaError::OperatingSystemError);
        };

        if let AsyncStatus::Completed { result } = &*status {
            #[allow(clippy::let_unit_value)]
            let () = (*result)?;

            std::mem::drop(status);

            // the computation has completed, so the result is available on any stream
            return Ok(Async {
                _stream: PhantomData::<&'stream_new Stream>,
                event: self.event,
                value: self.value,
                status: self.status,
            });
        }

        std::mem::drop(status);

        stream.wait_event(&self.event, StreamWaitEventFlags::DEFAULT)?;
        self.event.record(stream)?;

        // add a new waker callback since the waker may have received a spurious
        //  wake-up from when the computation completed on the original stream
        let waker_callback = self.status.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut status) = waker_callback.lock() {
                if let AsyncStatus::Processing { waker, .. } = &mut *status {
                    if let Some(waker) = waker.take() {
                        waker.wake();
                    }
                }
            }
        }))?;

        Ok(Async {
            _stream: PhantomData::<&'stream_new Stream>,
            event: self.event,
            value: self.value,
            status: self.status,
        })
    }
}

#[cfg(feature = "host")]
impl<'stream, T, C> Async<'stream, T, C> {
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
impl<'stream, T, C> IntoFuture for Async<'stream, T, C> {
    type Output = CudaResult<T>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let mut wrapper = Some(self);

        std::future::poll_fn(move |cx| {
            let poll = match &wrapper {
                #[allow(clippy::option_if_let_else)]
                Some(Async {
                    status: status_mutex,
                    ..
                }) => match status_mutex.lock() {
                    Ok(mut status_guard) => match &mut *status_guard {
                        AsyncStatus::Completed { result: Ok(()) } => Poll::Ready(Ok(())),
                        AsyncStatus::Completed { result: Err(err) } => Poll::Ready(Err(*err)),
                        AsyncStatus::Processing { waker, .. } => {
                            *waker = Some(cx.waker().clone());
                            Poll::Pending
                        },
                    },
                    Err(_) => Poll::Ready(Err(CudaError::OperatingSystemError)),
                },
                None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
            };

            match poll {
                Poll::Ready(Ok(())) => match wrapper.take() {
                    Some(Async { value, .. }) => Poll::Ready(Ok(value)),
                    None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
                },
                Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
                Poll::Pending => Poll::Pending,
            }
        })
    }
}
