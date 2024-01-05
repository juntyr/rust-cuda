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
pub trait Completion<T: ?Sized + BorrowMut<Self::Completed>>: sealed::Sealed {
    type Completed: ?Sized;

    #[doc(hidden)]
    fn synchronize_on_drop(&self) -> bool;

    #[allow(clippy::missing_errors_doc)] // FIXME
    fn complete(self, completed: &mut Self::Completed) -> CudaResult<()>;
}
#[cfg(feature = "host")]
mod sealed {
    pub trait Sealed {}
}

#[cfg(feature = "host")]
impl<T: ?Sized> Completion<T> for NoCompletion {
    type Completed = T;

    #[inline]
    fn synchronize_on_drop(&self) -> bool {
        false
    }

    #[inline]
    fn complete(self, _completed: &mut Self::Completed) -> CudaResult<()> {
        Ok(())
    }
}
#[cfg(feature = "host")]
impl sealed::Sealed for NoCompletion {}

#[cfg(feature = "host")]
impl<'a, T: ?Sized + BorrowMut<B>, B: ?Sized> Completion<T> for CompletionFnMut<'a, B> {
    type Completed = B;

    #[inline]
    fn synchronize_on_drop(&self) -> bool {
        true
    }

    #[inline]
    fn complete(self, completed: &mut Self::Completed) -> CudaResult<()> {
        (self)(completed)
    }
}
#[cfg(feature = "host")]
impl<'a, T: ?Sized> sealed::Sealed for CompletionFnMut<'a, T> {}

#[cfg(feature = "host")]
impl<T: ?Sized + BorrowMut<C::Completed>, C: Completion<T>> Completion<T> for Option<C> {
    type Completed = C::Completed;

    #[inline]
    fn synchronize_on_drop(&self) -> bool {
        self.as_ref().map_or(false, Completion::synchronize_on_drop)
    }

    #[inline]
    fn complete(self, completed: &mut Self::Completed) -> CudaResult<()> {
        self.map_or(Ok(()), |completion| completion.complete(completed))
    }
}
#[cfg(feature = "host")]
impl<C> sealed::Sealed for Option<C> {}

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
        let event = CudaDropWrapper::from(Event::new(EventFlags::DISABLE_TIMING)?);

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
    /// Calling `synchronize` after the computation has completed, e.g. after
    /// calling [`rustacuda::stream::Stream::synchronize`], should be very
    /// cheap.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn synchronize(self) -> CudaResult<T> {
        let (mut value, status) = self.destructure_into_parts();

        let (receiver, completion) = match status {
            AsyncStatus::Completed { result } => return result.map(|()| value),
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

        completion.complete(value.borrow_mut())?;

        Ok(value)
    }

    /// Moves the asynchronous data move to a different [`Stream`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn move_to_stream<'stream_new>(
        self,
        stream: &'stream_new Stream,
    ) -> CudaResult<Async<'a, 'stream_new, T, C>> {
        let (mut value, status) = self.destructure_into_parts();

        let (receiver, completion, event) = match status {
            AsyncStatus::Completed { .. } => {
                return Ok(Async {
                    _stream: PhantomData::<&'stream_new Stream>,
                    value,
                    status,
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
                    value,
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

        completion.complete(value.borrow_mut())?;

        Ok(Async {
            _stream: PhantomData::<&'stream_new Stream>,
            value,
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
        let (value, status) = self.destructure_into_parts();

        match status {
            AsyncStatus::Completed { result: Ok(()) } => Ok((value, None)),
            AsyncStatus::Completed { result: Err(err) } => Err(err),
            AsyncStatus::Processing {
                receiver: _,
                completion,
                event: _,
                _capture,
            } => Ok((value, Some(completion))),
        }
    }

    pub const fn as_ref(&self) -> AsyncProj<'_, 'stream, &T> {
        AsyncProj::new(&self.value)
    }

    pub fn as_mut(&mut self) -> AsyncProj<'_, 'stream, &mut T> {
        AsyncProj::new(&mut self.value)
    }

    #[must_use]
    fn destructure_into_parts(self) -> (T, AsyncStatus<'a, T, C>) {
        let this = std::mem::ManuallyDrop::new(self);

        // Safety: we destructure self into its droppable components,
        //         value and status, without dropping self itself
        unsafe { (std::ptr::read(&this.value), (std::ptr::read(&this.status))) }
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> Drop for Async<'a, 'stream, T, C> {
    fn drop(&mut self) {
        let AsyncStatus::Processing {
            receiver,
            completion,
            event: _,
            _capture,
        } = std::mem::replace(&mut self.status, AsyncStatus::Completed { result: Ok(()) })
        else {
            return;
        };

        if completion.synchronize_on_drop() && receiver.recv() == Ok(Ok(())) {
            let _ = completion.complete(self.value.borrow_mut());
        }
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
        let (value, status) = self.destructure_into_parts();

        let (completion, status): (Option<C>, AsyncStatus<'a, T, NoCompletion>) = match status {
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
            value: Some(value),
            completion,
            status,
        }
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: BorrowMut<C::Completed>, C: Completion<T>> Drop
    for AsyncFuture<'a, 'stream, T, C>
{
    fn drop(&mut self) {
        let Some(mut value) = self.value.take() else {
            return;
        };

        let AsyncStatus::Processing {
            receiver,
            completion: NoCompletion,
            event: _,
            _capture,
        } = std::mem::replace(&mut self.status, AsyncStatus::Completed { result: Ok(()) })
        else {
            return;
        };

        let Some(completion) = self.completion.take() else {
            return;
        };

        if completion.synchronize_on_drop() && receiver.recv() == Ok(Ok(())) {
            let _ = completion.complete(value.borrow_mut());
        }
    }
}

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
#[derive(Copy, Clone)]
pub struct AsyncProj<'a, 'stream, T: 'a> {
    _capture: PhantomData<&'a ()>,
    _stream: PhantomData<&'stream Stream>,
    value: T,
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: 'a> AsyncProj<'a, 'stream, T> {
    #[must_use]
    pub(crate) const fn new(value: T) -> Self {
        Self {
            _capture: PhantomData::<&'a ()>,
            _stream: PhantomData::<&'stream Stream>,
            value,
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
    pub(crate) unsafe fn unwrap_unchecked(self) -> T {
        self.value
    }
}
