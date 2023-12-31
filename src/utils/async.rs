#[cfg(feature = "host")]
use std::{future::Future, future::IntoFuture, marker::PhantomData, task::Poll};

#[cfg(feature = "host")]
use rustacuda::{
    error::CudaError, error::CudaResult, event::Event, event::EventFlags, stream::Stream,
    stream::StreamWaitEventFlags,
};

#[cfg(feature = "host")]
use crate::host::CudaDropWrapper;

#[cfg(feature = "host")]
pub struct Async<'stream, T, C> {
    _stream: PhantomData<&'stream Stream>,
    value: T,
    status: AsyncStatus<T, C>,
}

#[cfg(feature = "host")]
enum AsyncStatus<T, C> {
    #[allow(clippy::type_complexity)]
    Processing {
        receiver: oneshot::Receiver<CudaResult<()>>,
        capture: C,
        on_completion: Box<dyn FnOnce(&mut T, C) -> CudaResult<()>>,
        event: CudaDropWrapper<Event>,
    },
    Completed {
        result: CudaResult<()>,
    },
}

// TODO: completion is NOT allowed to make any cuda calls
#[cfg(feature = "host")]
impl<'stream, T, C> Async<'stream, T, C> {
    /// Wraps a `value` which is ready on `stream`.
    #[must_use]
    pub const fn ready(value: T, stream: &'stream Stream) -> Self {
        let _ = stream;

        Self {
            _stream: PhantomData::<&'stream Stream>,
            value,
            status: AsyncStatus::Completed { result: Ok(()) },
        }
    }

    /// Wraps a still-pending `value` which is being computed on `stream`
    /// such that its computation can be synchronised on.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn pending(
        value: T,
        stream: &'stream Stream,
        capture: C,
        on_completion: impl FnOnce(&mut T, C) -> CudaResult<()> + 'static,
    ) -> CudaResult<Self> {
        let event = CudaDropWrapper::from(Event::new(
            EventFlags::DISABLE_TIMING | EventFlags::BLOCKING_SYNC,
        )?);

        let (sender, receiver) = oneshot::channel();

        stream.add_callback(Box::new(|result| std::mem::drop(sender.send(result))))?;
        event.record(stream)?;

        Ok(Self {
            _stream: PhantomData::<&'stream Stream>,
            value,
            status: AsyncStatus::Processing {
                capture,
                receiver,
                on_completion: Box::new(on_completion),
                event,
            },
        })
    }

    /// Synchronises on this computation to block until it has completed and
    /// the inner value can be safely returned and again be used in synchronous
    /// operations.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn synchronize(mut self) -> CudaResult<T> {
        let (receiver, capture, on_completion) = match self.status {
            AsyncStatus::Completed { result } => return result.map(|()| self.value),
            AsyncStatus::Processing {
                receiver,
                capture,
                on_completion,
                event: _,
            } => (receiver, capture, on_completion),
        };

        match receiver.recv() {
            Ok(Ok(())) => (),
            Ok(Err(err)) => return Err(err),
            Err(oneshot::RecvError) => return Err(CudaError::AlreadyAcquired),
        }

        on_completion(&mut self.value, capture)?;

        Ok(self.value)
    }

    /// Moves the asynchronous data move to a different [`Stream`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn move_to_stream<'stream_new>(
        mut self,
        stream: &'stream_new Stream,
    ) -> CudaResult<Async<'stream_new, T, C>> {
        let (receiver, capture, on_completion, event) = match self.status {
            AsyncStatus::Completed { .. } => {
                return Ok(Async {
                    _stream: PhantomData::<&'stream_new Stream>,
                    value: self.value,
                    status: self.status,
                })
            },
            AsyncStatus::Processing {
                receiver,
                capture,
                on_completion,
                event,
            } => (receiver, capture, on_completion, event),
        };

        match receiver.try_recv() {
            Ok(Ok(())) => (),
            Ok(Err(err)) => return Err(err),
            Err(oneshot::TryRecvError::Empty) => {
                stream.wait_event(&event, StreamWaitEventFlags::DEFAULT)?;

                return Ok(Async {
                    _stream: PhantomData::<&'stream_new Stream>,
                    value: self.value,
                    status: AsyncStatus::Processing {
                        receiver,
                        capture,
                        on_completion,
                        event,
                    },
                });
            },
            Err(oneshot::TryRecvError::Disconnected) => return Err(CudaError::AlreadyAcquired),
        };

        on_completion(&mut self.value, capture)?;

        Ok(Async {
            _stream: PhantomData::<&'stream_new Stream>,
            value: self.value,
            status: AsyncStatus::Completed { result: Ok(()) },
        })
    }

    #[allow(clippy::missing_errors_doc)] // FIXME
    #[allow(clippy::type_complexity)] // FIXME
    /// # Safety
    ///
    /// The returned inner value of type `T` may not yet have completed its
    /// asynchronous work and may thus be in an inconsistent state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub unsafe fn unwrap_unchecked(
        self,
    ) -> CudaResult<(T, Option<(C, Box<dyn FnOnce(&mut T, C) -> CudaResult<()>>)>)> {
        match self.status {
            AsyncStatus::Completed { result: Ok(()) } => Ok((self.value, None)),
            AsyncStatus::Completed { result: Err(err) } => Err(err),
            AsyncStatus::Processing {
                receiver: _,
                capture,
                on_completion,
                event: _,
            } => Ok((self.value, Some((capture, on_completion)))),
        }
    }
}

#[cfg(feature = "host")]
struct AsyncFuture<'stream, T, C> {
    _stream: PhantomData<&'stream Stream>,
    value: Option<T>,
    #[allow(clippy::type_complexity)]
    capture_on_completion: Option<(C, Box<dyn FnOnce(&mut T, C) -> CudaResult<()> + 'static>)>,
    status: AsyncStatus<T, ()>,
}

#[cfg(feature = "host")]
impl<'stream, T, C> Future for AsyncFuture<'stream, T, C> {
    type Output = CudaResult<T>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> Poll<Self::Output> {
        // Safety: this function does not move out of `this`
        let this = unsafe { self.get_unchecked_mut() };

        match &mut this.status {
            AsyncStatus::Processing {
                receiver,
                capture: (),
                on_completion: _,
                event: _,
            } => match std::pin::Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(Ok(()))) => (),
                Poll::Ready(Ok(Err(err))) => return Poll::Ready(Err(err)),
                Poll::Ready(Err(oneshot::RecvError)) => {
                    return Poll::Ready(Err(CudaError::AlreadyAcquired))
                },
                Poll::Pending => return Poll::Pending,
            },
            AsyncStatus::Completed { result: Ok(()) } => (),
            AsyncStatus::Completed { result: Err(err) } => return Poll::Ready(Err(*err)),
        }

        let Some(mut value) = this.value.take() else {
            return Poll::Ready(Err(CudaError::AlreadyAcquired));
        };

        if let Some((capture, on_completion)) = this.capture_on_completion.take() {
            on_completion(&mut value, capture)?;
        }

        Poll::Ready(Ok(value))
    }
}

#[cfg(feature = "host")]
impl<'stream, T, C> IntoFuture for Async<'stream, T, C> {
    type Output = CudaResult<T>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let (capture_on_completion, status) = match self.status {
            AsyncStatus::Completed { result } => (None, AsyncStatus::Completed { result }),
            AsyncStatus::Processing {
                receiver,
                capture,
                on_completion,
                event,
            } => (
                Some((capture, on_completion)),
                AsyncStatus::Processing {
                    receiver,
                    capture: (),
                    on_completion: Box::new(|_self, ()| Ok(())),
                    event,
                },
            ),
        };

        AsyncFuture {
            _stream: PhantomData::<&'stream Stream>,
            value: Some(self.value),
            capture_on_completion,
            status,
        }
    }
}
