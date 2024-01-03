#[cfg(feature = "host")]
use std::{borrow::BorrowMut, future::Future, future::IntoFuture, marker::PhantomData, task::Poll};

#[cfg(feature = "host")]
use rustacuda::{
    error::CudaError, error::CudaResult, event::Event, event::EventFlags, stream::Stream,
    stream::StreamWaitEventFlags,
};

#[cfg(feature = "host")]
use crate::host::CudaDropWrapper;

#[cfg(feature = "host")]
pub struct NoCompletion;
#[cfg(feature = "host")]
pub type CompletionFnMut<'a, T> = Box<dyn FnOnce(&mut T) -> CudaResult<()> + 'a>;

#[cfg(feature = "host")]
pub trait Completion<T: BorrowMut<Self::Completed>>: sealed::Sealed {
    type Completed: ?Sized;

    #[allow(clippy::missing_errors_doc)] // FIXME
    fn complete(self, completed: &mut Self::Completed) -> CudaResult<()>;
}
#[cfg(feature = "host")]
mod sealed {
    pub trait Sealed {}
}

#[cfg(feature = "host")]
impl<T> Completion<T> for NoCompletion {
    type Completed = T;

    fn complete(self, _completed: &mut Self::Completed) -> CudaResult<()> {
        Ok(())
    }
}
#[cfg(feature = "host")]
impl sealed::Sealed for NoCompletion {}

#[cfg(feature = "host")]
impl<'a, T: BorrowMut<B>, B: ?Sized> Completion<T> for CompletionFnMut<'a, B> {
    type Completed = B;

    fn complete(self, completed: &mut Self::Completed) -> CudaResult<()> {
        (self)(completed)
    }
}
#[cfg(feature = "host")]
impl<'a, T: ?Sized> sealed::Sealed for CompletionFnMut<'a, T> {}

#[cfg(feature = "host")]
pub struct Async<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T> = NoCompletion> {
    _stream: PhantomData<&'stream Stream>,
    value: T,
    status: AsyncStatus<'a, T, C>,
    _capture: PhantomData<&'a ()>,
}

#[cfg(feature = "host")]
enum AsyncStatus<'a, T: BorrowMut<C::Completed>, C: Completion<T>> {
    #[allow(clippy::type_complexity)]
    Processing {
        receiver: oneshot::Receiver<CudaResult<()>>,
        completion: C,
        event: CudaDropWrapper<Event>,
        _capture: PhantomData<&'a T>,
    },
    Completed {
        result: CudaResult<()>,
    },
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> Async<'a, 'stream, T, C> {
    /// Wraps a `value` which is ready on `stream`.
    #[must_use]
    pub const fn ready(value: T, stream: &'stream Stream) -> Self {
        let _ = stream;

        Self {
            _stream: PhantomData::<&'stream Stream>,
            value,
            status: AsyncStatus::Completed { result: Ok(()) },
            _capture: PhantomData::<&'a ()>,
        }
    }

    /// Wraps a still-pending `value` which is being computed on `stream`
    /// such that its computation can be synchronised on.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn pending(value: T, stream: &'stream Stream, completion: C) -> CudaResult<Self> {
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
                receiver,
                completion,
                event,
                _capture: PhantomData::<&'a T>,
            },
            _capture: PhantomData::<&'a ()>,
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
        let (receiver, completion) = match self.status {
            AsyncStatus::Completed { result } => return result.map(|()| self.value),
            AsyncStatus::Processing {
                receiver,
                completion,
                event: _,
                _capture,
            } => (receiver, completion),
        };

        match receiver.recv() {
            Ok(Ok(())) => (),
            Ok(Err(err)) => return Err(err),
            Err(oneshot::RecvError) => return Err(CudaError::AlreadyAcquired),
        }

        completion.complete(self.value.borrow_mut())?;

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
    ) -> CudaResult<Async<'a, 'stream_new, T, C>> {
        let (receiver, completion, event) = match self.status {
            AsyncStatus::Completed { .. } => {
                return Ok(Async {
                    _stream: PhantomData::<&'stream_new Stream>,
                    value: self.value,
                    status: self.status,
                    _capture: PhantomData::<&'a ()>,
                })
            },
            AsyncStatus::Processing {
                receiver,
                completion,
                event,
                _capture,
            } => (receiver, completion, event),
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
                        completion,
                        event,
                        _capture: PhantomData::<&'a T>,
                    },
                    _capture: PhantomData::<&'a ()>,
                });
            },
            Err(oneshot::TryRecvError::Disconnected) => return Err(CudaError::AlreadyAcquired),
        };

        completion.complete(self.value.borrow_mut())?;

        Ok(Async {
            _stream: PhantomData::<&'stream_new Stream>,
            value: self.value,
            status: AsyncStatus::Completed { result: Ok(()) },
            _capture: PhantomData::<&'a ()>,
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
    pub unsafe fn unwrap_unchecked(self) -> CudaResult<(T, Option<C>)> {
        match self.status {
            AsyncStatus::Completed { result: Ok(()) } => Ok((self.value, None)),
            AsyncStatus::Completed { result: Err(err) } => Err(err),
            AsyncStatus::Processing {
                receiver: _,
                completion,
                event: _,
                _capture,
            } => Ok((self.value, Some(completion))),
        }
    }

    /// # Safety
    ///
    /// The returned reference to the inner value of type `T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub const unsafe fn unwrap_ref_unchecked(&self) -> &T {
        &self.value
    }

    /// # Safety
    ///
    /// The returned reference to the inner value of type `T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub unsafe fn unwrap_mut_unchecked(&mut self) -> &mut T {
        &mut self.value
    }
}

#[cfg(feature = "host")]
struct AsyncFuture<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> {
    _stream: PhantomData<&'stream Stream>,
    value: Option<T>,
    completion: Option<C>,
    status: AsyncStatus<'a, T, NoCompletion>,
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> Future
    for AsyncFuture<'a, 'stream, T, C>
{
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
                completion: _,
                event: _,
                _capture,
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

        if let Some(completion) = this.completion.take() {
            completion.complete(value.borrow_mut())?;
        }

        Poll::Ready(Ok(value))
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> IntoFuture
    for Async<'a, 'stream, T, C>
{
    type Output = CudaResult<T>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let (completion, status): (Option<C>, AsyncStatus<'a, T, NoCompletion>) = match self.status
        {
            AsyncStatus::Completed { result } => {
                (None, AsyncStatus::Completed::<T, NoCompletion> { result })
            },
            AsyncStatus::Processing {
                receiver,
                completion,
                event,
                _capture,
            } => (
                Some(completion),
                AsyncStatus::Processing::<T, NoCompletion> {
                    receiver,
                    completion: NoCompletion,
                    event,
                    _capture: PhantomData::<&'a T>,
                },
            ),
        };

        AsyncFuture {
            _stream: PhantomData::<&'stream Stream>,
            value: Some(self.value),
            completion,
            status,
        }
    }
}
