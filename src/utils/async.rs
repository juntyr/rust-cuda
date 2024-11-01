#[cfg(feature = "host")]
use std::{borrow::BorrowMut, future::Future, future::IntoFuture, marker::PhantomData, task::Poll};

#[cfg(feature = "host")]
use rustacuda::{
    error::CudaError, error::CudaResult, event::Event, event::EventFlags,
    stream::StreamWaitEventFlags,
};

#[cfg(feature = "host")]
use crate::host::{CudaDropWrapper, Stream};

#[cfg(feature = "host")]
pub struct NoCompletion;
#[cfg(feature = "host")]
pub type CompletionFnMut<'a, T> = Box<dyn FnOnce(&mut T) -> CudaResult<()> + 'a>;

#[cfg(feature = "host")]
pub trait Completion<T: ?Sized + BorrowMut<Self::Completed>>: sealed::Sealed {
    type Completed: ?Sized;

    fn no_op() -> Self;

    #[doc(hidden)]
    fn synchronize_on_drop(&self) -> bool;

    #[expect(clippy::missing_errors_doc)] // FIXME
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
    fn no_op() -> Self {
        Self
    }

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
    fn no_op() -> Self {
        Box::new(|_value| Ok(()))
    }

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
    fn no_op() -> Self {
        None
    }

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
    stream: Stream<'stream>,
    value: T,
    status: AsyncStatus<'a, T, C>,
    _capture: PhantomData<&'a ()>,
}

#[cfg(feature = "host")]
enum AsyncStatus<'a, T: BorrowMut<C::Completed>, C: Completion<T>> {
    Processing {
        receiver: oneshot::Receiver<CudaResult<()>>,
        completion: C,
        event: Option<CudaDropWrapper<Event>>,
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
    pub const fn ready(value: T, stream: Stream<'stream>) -> Self {
        Self {
            stream,
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
    pub fn pending(value: T, stream: Stream<'stream>, completion: C) -> CudaResult<Self> {
        let (sender, receiver) = oneshot::channel();
        stream.add_callback(Box::new(|result| std::mem::drop(sender.send(result))))?;

        Ok(Self {
            stream,
            value,
            status: AsyncStatus::Processing {
                receiver,
                completion,
                event: None,
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
        let (_stream, mut value, status) = self.destructure_into_parts();

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
    /// This method always adds a synchronisation barrier between the old and
    /// and the new [`Stream`] to ensure that any usages of this [`Async`]
    /// computations on the old [`Stream`] have completed before they can be
    /// used on the new one.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA.
    pub fn move_to_stream<'stream_new>(
        self,
        stream: Stream<'stream_new>,
    ) -> CudaResult<Async<'a, 'stream_new, T, C>> {
        let (old_stream, mut value, status) = self.destructure_into_parts();

        let completion = match status {
            AsyncStatus::Completed { result } => {
                result?;
                C::no_op()
            },
            AsyncStatus::Processing {
                receiver,
                completion,
                event: _,
                _capture,
            } => match receiver.try_recv() {
                Ok(Ok(())) => {
                    completion.complete(value.borrow_mut())?;
                    C::no_op()
                },
                Ok(Err(err)) => return Err(err),
                Err(oneshot::TryRecvError::Empty) => completion,
                Err(oneshot::TryRecvError::Disconnected) => return Err(CudaError::AlreadyAcquired),
            },
        };

        let event = CudaDropWrapper::from(Event::new(EventFlags::DISABLE_TIMING)?);
        event.record(&old_stream)?;
        stream.wait_event(&event, StreamWaitEventFlags::DEFAULT)?;

        let (sender, receiver) = oneshot::channel();
        stream.add_callback(Box::new(|result| std::mem::drop(sender.send(result))))?;

        Ok(Async {
            stream,
            value,
            status: AsyncStatus::Processing {
                receiver,
                completion,
                event: Some(event),
                _capture: PhantomData::<&'a T>,
            },
            _capture: PhantomData::<&'a ()>,
        })
    }

    #[expect(clippy::missing_errors_doc)] // FIXME
    /// # Safety
    ///
    /// The returned inner value of type `T` may not yet have completed its
    /// asynchronous work and may thus be in an inconsistent state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub unsafe fn unwrap_unchecked(self) -> CudaResult<(T, Option<C>)> {
        let (_stream, value, status) = self.destructure_into_parts();

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
        // Safety: this projection captures this async
        unsafe { AsyncProj::new(&self.value, None) }
    }

    pub fn as_mut(&mut self) -> AsyncProj<'_, 'stream, &mut T> {
        // Safety: this projection captures this async
        unsafe {
            AsyncProj::new(
                &mut self.value,
                Some(Box::new(|| {
                    let completion = match &mut self.status {
                        AsyncStatus::Completed { result } => {
                            (*result)?;
                            C::no_op()
                        },
                        AsyncStatus::Processing {
                            receiver: _,
                            completion,
                            event: _,
                            _capture,
                        } => std::mem::replace(completion, C::no_op()),
                    };

                    let event = CudaDropWrapper::from(Event::new(EventFlags::DISABLE_TIMING)?);

                    let (sender, receiver) = oneshot::channel();

                    self.stream
                        .add_callback(Box::new(|result| std::mem::drop(sender.send(result))))?;
                    event.record(&self.stream)?;

                    self.status = AsyncStatus::Processing {
                        receiver,
                        completion,
                        event: Some(event),
                        _capture: PhantomData::<&'a T>,
                    };

                    Ok(())
                })),
            )
        }
    }

    #[must_use]
    fn destructure_into_parts(self) -> (Stream<'stream>, T, AsyncStatus<'a, T, C>) {
        let this = std::mem::ManuallyDrop::new(self);

        let stream = this.stream;
        // Safety: this is never dropped and this.value only read once
        let value = unsafe { std::ptr::read(&this.value) };
        // Safety: this is never dropped and this.status only read once
        let status = unsafe { std::ptr::read(&this.status) };

        (stream, value, status)
    }
}

#[cfg(feature = "host")]
impl<
        'a,
        'stream,
        T: crate::safety::PortableBitSemantics + const_type_layout::TypeGraphLayout,
        C: Completion<crate::host::HostAndDeviceConstRef<'a, T>>,
    > Async<'a, 'stream, crate::host::HostAndDeviceConstRef<'a, T>, C>
where
    crate::host::HostAndDeviceConstRef<'a, T>: BorrowMut<C::Completed>,
{
    pub const fn extract_ref(
        &self,
    ) -> AsyncProj<'_, 'stream, crate::host::HostAndDeviceConstRef<'_, T>> {
        // Safety: this projection captures this async
        unsafe { AsyncProj::new(self.value.as_ref(), None) }
    }
}

#[cfg(feature = "host")]
impl<
        'a,
        'stream,
        T: crate::safety::PortableBitSemantics + const_type_layout::TypeGraphLayout,
        C: Completion<crate::host::HostAndDeviceMutRef<'a, T>>,
    > Async<'a, 'stream, crate::host::HostAndDeviceMutRef<'a, T>, C>
where
    crate::host::HostAndDeviceMutRef<'a, T>: BorrowMut<C::Completed>,
{
    pub fn extract_ref(&self) -> AsyncProj<'_, 'stream, crate::host::HostAndDeviceConstRef<'_, T>> {
        // Safety: this projection captures this async
        unsafe { AsyncProj::new(self.value.as_ref(), None) }
    }

    pub fn extract_mut(
        &mut self,
    ) -> AsyncProj<'_, 'stream, crate::host::HostAndDeviceMutRef<'_, T>> {
        // Safety: this projection captures this async
        unsafe {
            AsyncProj::new(
                self.value.as_mut(),
                Some(Box::new(|| {
                    let completion = match &mut self.status {
                        AsyncStatus::Completed { result } => {
                            (*result)?;
                            C::no_op()
                        },
                        AsyncStatus::Processing {
                            receiver: _,
                            completion,
                            event: _,
                            _capture,
                        } => std::mem::replace(completion, C::no_op()),
                    };

                    let event = CudaDropWrapper::from(Event::new(EventFlags::DISABLE_TIMING)?);

                    let (sender, receiver) = oneshot::channel();

                    self.stream
                        .add_callback(Box::new(|result| std::mem::drop(sender.send(result))))?;
                    event.record(&self.stream)?;

                    self.status = AsyncStatus::Processing {
                        receiver,
                        completion,
                        event: Some(event),
                        _capture: PhantomData,
                    };

                    Ok(())
                })),
            )
        }
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
    _stream: PhantomData<Stream<'stream>>,
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
        let (_stream, value, status) = self.destructure_into_parts();

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
            _stream: PhantomData::<Stream<'stream>>,
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
#[expect(clippy::module_name_repetitions)]
pub struct AsyncProj<'a, 'stream, T: 'a> {
    _capture: PhantomData<&'a ()>,
    _stream: PhantomData<Stream<'stream>>,
    value: T,
    use_callback: Option<Box<dyn FnMut() -> CudaResult<()> + 'a>>,
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: 'a> AsyncProj<'a, 'stream, T> {
    #[must_use]
    /// # Safety
    ///
    /// This projection must either capture an existing [`Async`] or come from
    /// a source that ensures that the projected value can never (async) move
    /// to a different [`Stream`].
    pub(crate) const unsafe fn new(
        value: T,
        use_callback: Option<Box<dyn FnMut() -> CudaResult<()> + 'a>>,
    ) -> Self {
        Self {
            _capture: PhantomData::<&'a ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value,
            use_callback,
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

    #[expect(clippy::type_complexity)]
    /// # Safety
    ///
    /// The returned reference to the inner value of type `T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub(crate) unsafe fn unwrap_unchecked_with_use(
        self,
    ) -> (T, Option<Box<dyn FnMut() -> CudaResult<()> + 'a>>) {
        (self.value, self.use_callback)
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: 'a> AsyncProj<'a, 'stream, T> {
    #[must_use]
    pub const fn proj_ref<'b>(&'b self) -> AsyncProj<'b, 'stream, &'b T>
    where
        'a: 'b,
    {
        AsyncProj {
            _capture: PhantomData::<&'b ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value: &self.value,
            use_callback: None,
        }
    }

    #[must_use]
    pub fn proj_mut<'b>(&'b mut self) -> AsyncProj<'b, 'stream, &'b mut T>
    where
        'a: 'b,
    {
        AsyncProj {
            _capture: PhantomData::<&'b ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value: &mut self.value,
            use_callback: self.use_callback.as_mut().map(|use_callback| {
                let use_callback: Box<dyn FnMut() -> CudaResult<()>> = Box::new(use_callback);
                use_callback
            }),
        }
    }

    pub(crate) fn record_mut_use(&mut self) -> CudaResult<()> {
        self.use_callback
            .as_mut()
            .map_or(Ok(()), |use_callback| use_callback())
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: 'a> AsyncProj<'a, 'stream, &'a T> {
    #[must_use]
    pub const fn as_ref<'b>(&'b self) -> AsyncProj<'b, 'stream, &'b T>
    where
        'a: 'b,
    {
        AsyncProj {
            _capture: PhantomData::<&'b ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value: self.value,
            use_callback: None,
        }
    }

    /// # Safety
    ///
    /// The returned reference to the inner value of type `&T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub(crate) const unsafe fn unwrap_ref_unchecked(&self) -> &T {
        self.value
    }
}

#[cfg(feature = "host")]
impl<'a, 'stream, T: 'a> AsyncProj<'a, 'stream, &'a mut T> {
    #[must_use]
    pub fn as_ref<'b>(&'b self) -> AsyncProj<'b, 'stream, &'b T>
    where
        'a: 'b,
    {
        AsyncProj {
            _capture: PhantomData::<&'b ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value: self.value,
            use_callback: None,
        }
    }

    #[must_use]
    pub fn as_mut<'b>(&'b mut self) -> AsyncProj<'b, 'stream, &'b mut T>
    where
        'a: 'b,
    {
        AsyncProj {
            _capture: PhantomData::<&'b ()>,
            _stream: PhantomData::<Stream<'stream>>,
            value: self.value,
            use_callback: self.use_callback.as_mut().map(|use_callback| {
                let use_callback: Box<dyn FnMut() -> CudaResult<()>> = Box::new(use_callback);
                use_callback
            }),
        }
    }

    #[expect(dead_code)] // FIXME
    /// # Safety
    ///
    /// The returned reference to the inner value of type `&T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub(crate) unsafe fn unwrap_ref_unchecked(&self) -> &T {
        self.value
    }

    #[expect(dead_code)] // FIXME
    /// # Safety
    ///
    /// The returned reference to the inner value of type `&T` may not yet have
    /// completed its asynchronous work and may thus be in an inconsistent
    /// state.
    ///
    /// This method must only be used to construct a larger asynchronous
    /// computation out of smaller ones that have all been submitted to the
    /// same [`Stream`].
    pub(crate) unsafe fn unwrap_mut_unchecked(&mut self) -> &mut T {
        self.value
    }
}
