#![deny(missing_debug_implementations, missing_docs)]
//! `tryiterator` is a crate for working with `Iterator`s that can produce error
//! values. It is inspired by `futures`'s `TryStream`/`TryStreamExt` and intended
//! to replace `fallible-iterator`.
//!
//! Most users will want to import the `prelude` module to get access to the
//! `TryIteratorExt` and `DoubleEndedTryIteratorExt` traits. The `TryIterator`
//! and `DoubleEndedTryIterator` traits should not need to be used directly
//! unless you are writing generic code.
//!
//! ## Motivation
//! `fallible-iterator` chose to represent the items yielded from an iterator
//! that can produce errors as `Result<Option<T>, E>`. This means that a
//! `FallibleIterator` is not an `Iterator`, and vice versa. The resulting type
//! mismatch makes `FallibleIterator`s more difficult to work with, as explicit
//! conversions to and from `Iterator` are required, and `FallibleIterator`s do
//! not automatically benefit from additions to the `Iterator` trait.
//!
//! `tryiterator` instead represents items yielded as `Option<Result<T, E>>`.
//! This is inspired by `TryFuture`/`TryStream` which are merely
//! `Future`/`Stream` implementations that yield `Result`s. Every `TryIterator`
//! is merely an `Iterator<Item=Result<T, E>>`. The `TryIteratorExt` trait then
//! provides methods that act explicitly on the success or failure cases,
//! and the methods already present on `Iterator` can be used to act on both
//! cases at once.
use core::fmt::Debug;
use core::iter::{FusedIterator, Iterator, Peekable};

/// Internal implementation details.
mod private_try_iterator {
    use super::Iterator;

    /// This trait exists to prevent any additional implementations of TryIterator.
    pub trait Sealed {}

    impl<I, T, E> Sealed for I where I: ?Sized + Iterator<Item = Result<T, E>> {}

    /// This trait exists to enable try_collect.
    pub trait Cast {
        type Ok;
        type Error;
        type Iter: Iterator<Item = Result<Self::Ok, Self::Error>>;
        fn cast(self) -> Self::Iter;
        fn cast_item_ref(item: &<Self::Iter as Iterator>::Item) -> &Result<Self::Ok, Self::Error>;
        fn cast_item_mut(
            item: &mut <Self::Iter as Iterator>::Item,
        ) -> &mut Result<Self::Ok, Self::Error>;
    }

    impl<I, T, E> Cast for I
    where
        I: Sized + Iterator<Item = Result<T, E>>,
    {
        type Ok = T;
        type Error = E;
        type Iter = I;
        fn cast(self) -> Self::Iter {
            self
        }
        fn cast_item_ref(item: &<Self::Iter as Iterator>::Item) -> &Result<Self::Ok, Self::Error> {
            item
        }
        fn cast_item_mut(
            item: &mut <Self::Iter as Iterator>::Item,
        ) -> &mut Result<Self::Ok, Self::Error> {
            item
        }
    }
}

/// A convenience for `Iterator`s that return `Result` values.
/// This trait should not be used except as a trait bound. Use
/// `TryIteratorExt`.
pub trait TryIterator: Iterator + private_try_iterator::Sealed {
    /// The type of successful values yielded by this iterator.
    type Ok;

    /// The type of failures yielded by this iterator.
    type Error;

    /// `Iterator::next` but with a fully specified type.
    /// Useful for building combinators.
    fn next(&mut self) -> Option<Result<Self::Ok, Self::Error>>;
    /// Prefer `TryIteratorExt::try_next` unless building combinators.
    fn try_next(&mut self) -> Result<Option<Self::Ok>, Self::Error>;
}

/// A convenience for `DoubleEndedIterator`s that return `Result` values.
/// This trait should not be used except as a trait bound. Use
/// `DoubleEndedTryIteratorExt`.
pub trait DoubleEndedTryIterator:
    TryIterator + DoubleEndedIterator + private_try_iterator::Sealed
{
    /// `DoubleEndedIterator::next` but with a fully specified type.
    /// Useful for building combinators.
    fn next_back(&mut self) -> Option<Result<Self::Ok, Self::Error>>;
    /// Prefer `DoubleEndedTryIteratorExt::try_next_back` unless building
    /// combinators.
    fn try_next_back(&mut self) -> Result<Option<Self::Ok>, Self::Error>;
}

impl<I, T, E> TryIterator for I
where
    I: ?Sized + Iterator<Item = Result<T, E>>,
{
    type Ok = T;
    type Error = E;

    fn next(&mut self) -> Option<Result<Self::Ok, Self::Error>> {
        Iterator::next(self)
    }
    fn try_next(&mut self) -> Result<Option<Self::Ok>, Self::Error> {
        Iterator::next(self).transpose()
    }
}

impl<I, T, E> DoubleEndedTryIterator for I
where
    I: ?Sized + DoubleEndedIterator<Item = Result<T, E>>,
{
    fn next_back(&mut self) -> Option<Result<Self::Ok, Self::Error>> {
        DoubleEndedIterator::next_back(self)
    }
    fn try_next_back(
        &mut self,
    ) -> Result<Option<<Self as TryIterator>::Ok>, <Self as TryIterator>::Error> {
        self.next_back().transpose()
    }
}

/// Adapters specific to `Result`-returning `Iterator`s.
pub trait TryIteratorExt: TryIterator {
    /// This returns a `Result<Option<T>, E>` rather than the
    /// `Option<Result<T, E>>` returned by `Iterator::next` for
    /// easy use with the `?` operator.
    fn try_next(&mut self) -> Result<Option<Self::Ok>, Self::Error> {
        TryIterator::try_next(self)
    }

    /// Map the success value using the provided closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(7u64), Err(42u64)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.map_ok(|x| x * 2);
    /// assert_eq!(new_iter.next(), Some(Ok(14)));
    /// assert_eq!(new_iter.next(), Some(Err(42)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn map_ok<T, F>(self, f: F) -> MapOk<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Ok) -> T,
    {
        MapOk::new(self, f)
    }

    /// Map the error value using the provided closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(7u64), Err(42u64)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.map_err(|x| x / 2);
    /// assert_eq!(new_iter.next(), Some(Ok(7)));
    /// assert_eq!(new_iter.next(), Some(Err(21)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn map_err<E, F>(self, f: F) -> MapErr<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Error) -> E,
    {
        MapErr::new(self, f)
    }

    /// Run an additional computation that can fail on the
    /// success value. Error values are passed through
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Ok(0), Err(7u64)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.and_then(|x| 42u64.checked_div(x).ok_or(0));
    /// assert_eq!(new_iter.next(), Some(Ok(21)));
    /// assert_eq!(new_iter.next(), Some(Err(0)));
    /// assert_eq!(new_iter.next(), Some(Err(7)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn and_then<T, F>(self, f: F) -> AndThen<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Ok) -> Result<T, Self::Error>,
    {
        AndThen::new(self, f)
    }

    /// Run an additional computation that can fail on the
    /// error value. Success values are passed through
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Err(0u64), Err(7)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.or_else(|x| 42u64.checked_div(x).ok_or(0));
    /// assert_eq!(new_iter.next(), Some(Ok(2)));
    /// assert_eq!(new_iter.next(), Some(Err(0)));
    /// assert_eq!(new_iter.next(), Some(Ok(6)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn or_else<E, F>(self, f: F) -> OrElse<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Error) -> Result<Self::Ok, E>,
    {
        OrElse::new(self, f)
    }

    /// Discard error values as if `Result::unwrap` were called on each item
    /// returned by the iterator.
    ///
    /// ```should_panic
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Err(0u64), Err(7)];
    /// let mut iter = values.into_iter().unwrap();
    /// assert_eq!(iter.next(), Some(2));
    /// // This panics!
    /// iter.next();
    /// ```
    fn unwrap(self) -> Unwrap<Self>
    where
        Self: Sized,
        Self::Error: Debug,
    {
        Unwrap::new(self)
    }

    /// Transform a `TryIterator` into a collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok::<_, u64>(2u64), Ok(0u64)];
    /// let iter = values.iter().copied();
    /// let values2 = iter.try_collect::<Vec<_>>();
    /// assert_eq!(values2, Ok(vec![2u64, 0u64]));
    /// let values3 = vec![Ok(2u64), Err(0u64)];
    /// let iter2 = values3.iter().copied();
    /// let values4 = iter2.try_collect::<Vec<_>>();
    /// assert_eq!(values4, Err(0));
    /// ```
    fn try_collect<B>(self) -> Result<B, <Self as TryIterator>::Error>
    where
        Self: Sized
            + private_try_iterator::Cast<
                Ok = <Self as TryIterator>::Ok,
                Error = <Self as TryIterator>::Error,
            >,
        B: FromIterator<<Self as TryIterator>::Ok>,
    {
        FromIterator::from_iter(self.cast())
    }

    /// Run an additional computation on the success value and choose to pass
    /// through or discard this item. Error values are passed through unchanged.
    /// If you want to modify the value, or this computation can fail, use
    /// `try_filter_map`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Ok(0u64), Err(7u64)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.try_filter(|x| 42u64.checked_div(*x).is_some());
    /// assert_eq!(new_iter.next(), Some(Ok(2)));
    /// // NB: The item 0 was discarded.
    /// assert_eq!(new_iter.next(), Some(Err(7)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn try_filter<F>(self, f: F) -> TryFilter<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Ok) -> bool,
    {
        TryFilter::new(self, f)
    }

    /// Run an additional computation that can fail on the
    /// success value and choose to produce a new value or
    /// discard this item. Error values are passed through
    /// unchanged.
    ///
    /// The difference between `try_filter_map` and `and_then`
    /// is that this function allows the closure to choose
    /// to discard items by returning `Ok(None)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Ok(0u64), Err(7u64)];
    /// let iter = values.into_iter();
    /// let mut new_iter = iter.try_filter_map(|x| Ok(42u64.checked_div(x)));
    /// assert_eq!(new_iter.next(), Some(Ok(21)));
    /// // NB: The item 0 was discarded.
    /// assert_eq!(new_iter.next(), Some(Err(7)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn try_filter_map<T, F>(self, f: F) -> TryFilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Ok) -> Result<Option<T>, Self::Error>,
    {
        TryFilterMap::new(self, f)
    }

    /// Flatten a `TryIterator` of `TryIterator` into a single continuous
    /// `TryIterator`. Error values are passed through
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::iter;
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values1 = vec![Ok(2u64), Ok(0u64), Err(7u64)];
    /// let iter = Box::new(values1.into_iter());
    /// let iter2 = Box::new(iter::empty()) as Box<dyn Iterator<Item = Result<u64, u64>>>;
    /// let values2 = vec![Ok(iter2), Err(12u64), Ok(iter)];
    /// let iter = values2.into_iter();
    /// let mut new_iter = iter.try_flatten();
    /// assert_eq!(new_iter.next(), Some(Err(12)));
    /// assert_eq!(new_iter.next(), Some(Ok(2)));
    /// assert_eq!(new_iter.next(), Some(Ok(0)));
    /// assert_eq!(new_iter.next(), Some(Err(7)));
    /// assert_eq!(new_iter.next(), None);
    /// ```
    fn try_flatten(self) -> TryFlatten<Self>
    where
        Self: Sized,
        Self::Ok: TryIterator,
        <Self::Ok as TryIterator>::Error: From<Self::Error>,
    {
        TryFlatten::new(self)
    }

    /// Attempt to execute an accumulating computation in the form of the provided
    /// closure on this `TryIterator`. The initial state is provided and the return
    /// value of the closure must be the same type. Errors in either the iterator
    /// or the closure will short-circuit the computation.
    ///
    /// NB: try_fold is already taken as a method on Iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok::<_, u64>(2u64), Ok(40)];
    /// let sum = values.into_iter()
    ///     .try_fold2(0u64, |mut sum, i| {
    ///         sum += i;
    ///         Ok::<_, u64>(sum)
    ///     });
    /// assert_eq!(sum, Ok(42));
    /// ```
    ///
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Err(40u64)];
    /// let sum = values.into_iter()
    ///     .try_fold2(0u64, |mut sum, i| {
    ///         sum += i;
    ///         Ok(sum)
    ///     });
    /// assert_eq!(sum, Err(40u64));
    /// ```
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok::<_, i64>(2u64), Ok(40u64)];
    /// let sum = values.into_iter()
    ///     .try_fold2(0u64, |_, _| Err(-42));
    /// assert_eq!(sum, Err(-42));
    /// ```
    fn try_fold2<B, F>(mut self, init: B, mut f: F) -> Result<B, Self::Error>
    where
        Self: Sized,
        F: FnMut(B, Self::Ok) -> Result<B, Self::Error>,
    {
        let mut accum = init;
        while let Some(x) = TryIterator::try_next(&mut self)? {
            accum = f(accum, x)?;
        }
        Ok(accum)
    }

    /// Calls a closure on each element of this `TryIterator`.
    /// Any error in the iterator will short-circuit the computation.
    /// If the closure can fail, or produces a value, use `try_fold2`.
    ///
    /// NB: try_for_each is already taken as a method on Iterator.
    ///
    /// # Examples
    /// ```
    /// use std::sync::mpsc::channel;
    /// use tryiterator::TryIteratorExt;
    ///
    /// let (tx, rx) = channel();
    /// let values = vec![Ok::<_, u64>(2u64), Ok(40)];
    /// values.into_iter()
    ///     .try_for_each2(move |i| tx.send(i).unwrap())
    ///     .unwrap();
    /// let v: Vec<_> = rx.iter().collect();
    /// assert_eq!(v, vec![2u64, 40]);
    /// ```
    /// ```
    /// use std::sync::mpsc::channel;
    /// use tryiterator::TryIteratorExt;
    ///
    /// let (tx, rx) = channel();
    /// let values = vec![Ok(2u64), Err(40u64), Ok(3)];
    /// let r = values.into_iter()
    ///     .try_for_each2(move |i| tx.send(i).unwrap());
    /// assert_eq!(r, Err(40));
    /// let v: Vec<_> = rx.iter().collect();
    /// assert_eq!(v, vec![2u64]);
    /// ```
    fn try_for_each2<F>(self, mut f: F) -> Result<(), Self::Error>
    where
        Self: Sized,
        F: FnMut(Self::Ok),
    {
        self.try_fold2((), |_, x| {
            f(x);
            Ok(())
        })
    }

    /// Skip elements from this `TryIterator` while the provided predicate
    /// returns true. Once the predicate returns false, that element and
    /// all future elements are passed through, and the predicate is not
    /// invoked again.
    ///
    /// If the predicate returns an error the iterator will cease, but errors
    /// from the original iterator are passed through.
    ///
    /// # Examples
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Err(7u64), Ok(0), Ok(1), Ok(2)];
    /// let mut iter = values.into_iter().try_skip_while(|x| {
    ///     match x {
    ///         0 => Err(0),
    ///         1 => Ok(false),
    ///         _ => Ok(true),
    ///     }
    /// });
    /// // Ok(2) is skipped by the predicate.
    /// // Err(7) is passed through by the combinator.
    /// assert_eq!(iter.next(), Some(Err(7)));
    /// // Ok(0) produces Err(0) in the predicate, which is returned.
    /// assert_eq!(iter.next(), Some(Err(0)));
    /// // Ok(1) causes the predicate to return false, and all further
    /// // elements are returned.
    /// assert_eq!(iter.next(), Some(Ok(1)));
    /// assert_eq!(iter.next(), Some(Ok(2)));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn try_skip_while<F>(self, f: F) -> TrySkipWhile<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Ok) -> Result<bool, Self::Error>,
    {
        TrySkipWhile::new(self, f)
    }

    /// Take elements from this `TryIterator` while the provided predicate
    /// returns true.  Once the predicate returns false, that element and
    /// all future elements are discarded, and the predicate is not invoked
    /// again.
    ///
    /// If the predicate returns an error the iterator will cease, but errors
    /// from the original iterator are passed through.
    ///
    /// # Examples
    /// ```
    /// use tryiterator::TryIteratorExt;
    ///
    /// let values = vec![Ok(2u64), Err(7u64), Ok(0), Ok(1), Ok(2)];
    /// let mut iter = values.into_iter().try_take_while(|x| {
    ///     match x {
    ///         0 => Err(0),
    ///         1 => Ok(false),
    ///         _ => Ok(true),
    ///     }
    /// });
    /// // Ok(2) is allowed by the predicate.
    /// assert_eq!(iter.next(), Some(Ok(2)));
    /// // Err(7) is passed through by the combinator.
    /// assert_eq!(iter.next(), Some(Err(7)));
    /// // Ok(0) produces Err(0) in the predicate, which is returned.
    /// assert_eq!(iter.next(), Some(Err(0)));
    /// // Ok(1) causes the predicate to return false, and no further
    /// // elements are returned.
    /// assert_eq!(iter.next(), None);
    /// ```
    fn try_take_while<F>(self, f: F) -> TryTakeWhile<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Ok) -> Result<bool, Self::Error>,
    {
        TryTakeWhile::new(self, f)
    }
}

/// Adapters specific to `Result`-returning `DoubleEndedIterator`s.
pub trait DoubleEndedTryIteratorExt: DoubleEndedTryIterator {
    /// This returns a `Result<Option<T>, E>` rather than the
    /// `Option<Result<T, E>>` returned by `DoubleEndedIterator::next_back` for
    /// easy use with the `?` operator.
    fn try_next_back(&mut self) -> Result<Option<Self::Ok>, Self::Error> {
        DoubleEndedTryIterator::try_next_back(self)
    }

    /// This is the reverse version of `TryIteratorExt::try_fold2`.
    ///
    /// NB: try_rfold is already taken as a method on DoubleEndedIterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tryiterator::DoubleEndedTryIteratorExt;
    ///
    /// let values = vec![Ok::<_, u64>(2u64), Ok(40)];
    /// let sum = values.into_iter()
    ///     .try_rfold2(0u64, |mut sum, i| {
    ///         sum += i;
    ///         Ok::<_, u64>(sum)
    ///     });
    /// assert_eq!(sum, Ok(42));
    /// ```
    ///
    /// ```
    /// use tryiterator::DoubleEndedTryIteratorExt;
    ///
    /// let values = vec![Err(20u64), Ok(2u64), Err(40)];
    /// let sum = values.into_iter()
    ///     .try_rfold2(0u64, |mut sum, i| {
    ///         sum += i;
    ///         Ok(sum)
    ///     });
    /// assert_eq!(sum, Err(40u64));
    /// ```
    /// ```
    /// use tryiterator::DoubleEndedTryIteratorExt;
    ///
    /// let values = vec![Ok::<_, i64>(2u64), Ok(40u64)];
    /// let sum = values.into_iter()
    ///     .try_rfold2(0u64, |_, _| Err(-42));
    /// assert_eq!(sum, Err(-42));
    /// ```
    fn try_rfold2<B, F>(mut self, init: B, mut f: F) -> Result<B, Self::Error>
    where
        Self: Sized,
        F: FnMut(B, Self::Ok) -> Result<B, Self::Error>,
    {
        let mut accum = init;
        while let Some(x) = DoubleEndedTryIterator::try_next_back(&mut self)? {
            accum = f(accum, x)?;
        }
        Ok(accum)
    }
}

/// An extension trait that adds methods to `Peekable` that are useful when the
/// underlying `Iterator` is a `TryIterator`.
pub trait TryPeekableExt {
    #[allow(missing_docs)]
    type I: TryIterator + private_try_iterator::Cast;

    /// This returns a `Result<Option<&T>, &E>` rather than the
    /// `Option<&Result<T, E>>` returned by `Peekable::peek`.
    ///
    /// ```
    /// use tryiterator::{TryIteratorExt, TryPeekableExt};
    ///
    /// let values = vec![Ok::<_, u64>(2u64), Ok(40)];
    /// let mut iter = values.into_iter().peekable();
    /// assert_eq!(iter.try_peek(), Ok(Some(&2)));
    /// ```
    fn try_peek(
        &mut self,
    ) -> Result<
        Option<&<<Self as TryPeekableExt>::I as TryIterator>::Ok>,
        &<<Self as TryPeekableExt>::I as TryIterator>::Error,
    >;

    /// This returns a `Result<Option<&mut T>, &mut E>` rather than the
    /// `Option<&mut Result<T, E>>` returned by `Peekable::peek_mut`.
    ///
    /// ```
    /// use tryiterator::{TryIteratorExt, TryPeekableExt};
    ///
    /// let values = vec![Ok::<_, u64>(2u64), Ok(40)];
    /// let mut iter = values.into_iter().peekable();
    /// assert_eq!(iter.try_peek_mut(), Ok(Some(&mut 2)));
    /// *iter.try_peek_mut().unwrap().unwrap() = 4;
    /// assert_eq!(iter.try_next(), Ok(Some(4)));
    /// ```
    fn try_peek_mut(
        &mut self,
    ) -> Result<
        Option<&mut <<Self as TryPeekableExt>::I as TryIterator>::Ok>,
        &mut <<Self as TryPeekableExt>::I as TryIterator>::Error,
    >;
}

impl<I> TryPeekableExt for Peekable<I>
where
    I: TryIterator
        + private_try_iterator::Cast<
            Iter = I,
            Ok = <I as TryIterator>::Ok,
            Error = <I as TryIterator>::Error,
        >,
{
    type I = I;
    fn try_peek(
        &mut self,
    ) -> Result<
        Option<&<<Self as TryPeekableExt>::I as TryIterator>::Ok>,
        &<<Self as TryPeekableExt>::I as TryIterator>::Error,
    > {
        self.peek()
            .map(I::cast_item_ref)
            .map(Result::as_ref)
            .transpose()
    }
    fn try_peek_mut(
        &mut self,
    ) -> Result<
        Option<&mut <<Self as TryPeekableExt>::I as TryIterator>::Ok>,
        &mut <<Self as TryPeekableExt>::I as TryIterator>::Error,
    > {
        self.peek_mut()
            .map(I::cast_item_mut)
            .map(Result::as_mut)
            .transpose()
    }
}

/// Iterator for the `TryIteratorExt::map_ok` method.
#[derive(Debug, Clone)]
pub struct MapOk<I, F> {
    inner: I,
    mutator: F,
}

impl<T, I, F> MapOk<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> T,
{
    fn new(inner: I, mutator: F) -> Self {
        MapOk { inner, mutator }
    }
}

impl<T, I, F> Iterator for MapOk<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> T,
{
    type Item = Result<T, <I as TryIterator>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(TryIterator::next(&mut self.inner)?.map(&mut self.mutator))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added or removed, so pass this through.
        self.inner.size_hint()
    }
}

impl<T, I, F> DoubleEndedIterator for MapOk<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> T,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(DoubleEndedTryIterator::next_back(&mut self.inner)?.map(&mut self.mutator))
    }
}

impl<T, I, F> FusedIterator for MapOk<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> T,
{
}

/// Iterator for the `TryIteratorExt::map_err` method.
#[derive(Debug, Clone)]
pub struct MapErr<I, F> {
    inner: I,
    mutator: F,
}

impl<E, I, F> MapErr<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> E,
{
    fn new(inner: I, mutator: F) -> Self {
        MapErr { inner, mutator }
    }
}

impl<E, I, F> Iterator for MapErr<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> E,
{
    type Item = Result<<I as TryIterator>::Ok, E>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(TryIterator::next(&mut self.inner)?.map_err(&mut self.mutator))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added or removed, so pass this through.
        self.inner.size_hint()
    }
}

impl<E, I, F> DoubleEndedIterator for MapErr<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> E,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(DoubleEndedTryIterator::next_back(&mut self.inner)?.map_err(&mut self.mutator))
    }
}

impl<E, I, F> FusedIterator for MapErr<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> E,
{
}

/// Iterator for the `TryIteratorExt::and_then` method.
#[derive(Debug, Clone)]
pub struct AndThen<I, F> {
    inner: I,
    mutator: F,
}

impl<T, I, F> AndThen<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<T, <I as TryIterator>::Error>,
{
    fn new(inner: I, mutator: F) -> Self {
        AndThen { inner, mutator }
    }
}

impl<T, I, F> Iterator for AndThen<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<T, <I as TryIterator>::Error>,
{
    type Item = Result<T, <I as TryIterator>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(TryIterator::next(&mut self.inner)?.and_then(&mut self.mutator))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added or removed, so pass this through.
        self.inner.size_hint()
    }
}

impl<T, I, F> DoubleEndedIterator for AndThen<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<T, <I as TryIterator>::Error>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(DoubleEndedTryIterator::next_back(&mut self.inner)?.and_then(&mut self.mutator))
    }
}

impl<T, I, F> FusedIterator for AndThen<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<T, <I as TryIterator>::Error>,
{
}

/// Iterator for the `TryIteratorExt::or_else` method.
#[derive(Debug, Clone)]
pub struct OrElse<I, F> {
    inner: I,
    mutator: F,
}

impl<E, I, F> OrElse<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> Result<<I as TryIterator>::Ok, E>,
{
    fn new(inner: I, mutator: F) -> Self {
        OrElse { inner, mutator }
    }
}

impl<E, I, F> Iterator for OrElse<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> Result<<I as TryIterator>::Ok, E>,
{
    type Item = Result<<I as TryIterator>::Ok, E>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(TryIterator::next(&mut self.inner)?.or_else(&mut self.mutator))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added or removed, so pass this through.
        self.inner.size_hint()
    }
}

impl<E, I, F> DoubleEndedIterator for OrElse<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> Result<<I as TryIterator>::Ok, E>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(DoubleEndedTryIterator::next_back(&mut self.inner)?.or_else(&mut self.mutator))
    }
}

impl<E, I, F> FusedIterator for OrElse<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(<I as TryIterator>::Error) -> Result<<I as TryIterator>::Ok, E>,
{
}

/// Iterator for the `TryIteratorExt::unwrap` method.
#[derive(Debug, Clone)]
pub struct Unwrap<I>
where
    I: TryIterator,
    I::Error: Debug,
{
    inner: I,
}

impl<I> Unwrap<I>
where
    I: TryIterator + Sized,
    I::Error: Debug,
{
    fn new(inner: I) -> Self {
        Unwrap { inner }
    }

    /// Acquires a mutable reference to the underlying iterator that this
    /// combinator is pulling from.
    ///
    /// Because `Unwrap` maintains no internal state it is safe to modify
    /// the underlying iterator.
    pub fn get_mut(&mut self) -> &mut I {
        &mut self.inner
    }
}

impl<I> Iterator for Unwrap<I>
where
    I: TryIterator + Sized,
    I::Error: Debug,
{
    type Item = <I as TryIterator>::Ok;

    fn next(&mut self) -> Option<Self::Item> {
        Some(TryIterator::next(&mut self.inner)?.unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added, so pass through the upper bound.
        // This could panic so we have no idea what the lower bound is.
        (0, self.inner.size_hint().1)
    }
}

impl<I> DoubleEndedIterator for Unwrap<I>
where
    I: DoubleEndedTryIterator + Sized,
    I::Error: Debug,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(DoubleEndedTryIterator::next_back(&mut self.inner)?.unwrap())
    }
}

impl<I> FusedIterator for Unwrap<I>
where
    I: TryIterator + FusedIterator + Sized,
    I::Error: Debug,
{
}

/// Iterator for the `TryIteratorExt::try_filter` method.
#[derive(Debug, Clone)]
pub struct TryFilter<I, F> {
    inner: I,
    filter: F,
}

impl<I, F> TryFilter<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&<I as TryIterator>::Ok) -> bool,
{
    fn new(inner: I, filter: F) -> Self {
        TryFilter { inner, filter }
    }
}

impl<I, F> Iterator for TryFilter<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&<I as TryIterator>::Ok) -> bool,
{
    type Item = Result<<I as TryIterator>::Ok, <I as TryIterator>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = match TryIterator::next(&mut self.inner)? {
                Ok(v) => v,
                Err(e) => break Some(Err(e)),
            };
            if (self.filter)(&item) {
                break Some(Ok(item));
            }
            // The closure discarded this item, continue with the next item.
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added, so pass through the upper bound.
        (0, self.inner.size_hint().1)
    }
}

impl<I, F> DoubleEndedIterator for TryFilter<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(&<I as TryIterator>::Ok) -> bool,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let item = match DoubleEndedTryIterator::next_back(&mut self.inner)? {
                Ok(v) => v,
                Err(e) => break Some(Err(e)),
            };
            if (self.filter)(&item) {
                break Some(Ok(item));
            }
            // The closure discarded this item, continue with the next item.
        }
    }
}

impl<I, F> FusedIterator for TryFilter<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(&<I as TryIterator>::Ok) -> bool,
{
}

/// Iterator for the `TryIteratorExt::try_future_map` method.
#[derive(Debug, Clone)]
pub struct TryFilterMap<I, F> {
    inner: I,
    mutator: F,
}

impl<T, I, F> TryFilterMap<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<Option<T>, <I as TryIterator>::Error>,
{
    fn new(inner: I, mutator: F) -> Self {
        TryFilterMap { inner, mutator }
    }
}

impl<T, I, F> Iterator for TryFilterMap<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<Option<T>, <I as TryIterator>::Error>,
{
    type Item = Result<T, <I as TryIterator>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = match TryIterator::next(&mut self.inner)? {
                Ok(v) => v,
                Err(e) => break Some(Err(e)),
            };
            if let Some(v) = (self.mutator)(item).transpose() {
                break Some(v);
            }
            // The closure discarded this item, continue with the next item.
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added, so pass through the upper bound.
        (0, self.inner.size_hint().1)
    }
}

impl<T, I, F> DoubleEndedIterator for TryFilterMap<I, F>
where
    I: DoubleEndedTryIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<Option<T>, <I as TryIterator>::Error>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let item = match DoubleEndedTryIterator::next_back(&mut self.inner)? {
                Ok(v) => v,
                Err(e) => break Some(Err(e)),
            };
            if let Some(v) = (self.mutator)(item).transpose() {
                break Some(v);
            }
            // The closure discarded this item, continue with the next item.
        }
    }
}

impl<T, I, F> FusedIterator for TryFilterMap<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(<I as TryIterator>::Ok) -> Result<Option<T>, <I as TryIterator>::Error>,
{
}

/// Iterator for the `TryIteratorExt::try_flatten` method.
#[derive(Debug, Clone)]
pub struct TryFlatten<I>
where
    I: TryIterator,
{
    inner: I,
    next: Option<I::Ok>,
}

impl<I> TryFlatten<I>
where
    I: TryIterator + Sized,
{
    fn new(inner: I) -> Self {
        TryFlatten { inner, next: None }
    }
}

impl<I> Iterator for TryFlatten<I>
where
    I: TryIterator + Sized,
    I::Ok: TryIterator,
    <I::Ok as TryIterator>::Error: From<I::Error>,
{
    type Item = Result<<I::Ok as TryIterator>::Ok, <I::Ok as TryIterator>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(next) = self.next.as_mut().and_then(TryIterator::next) {
                break Some(next.map_err(Into::into));
            }

            self.next = match TryIterator::next(&mut self.inner)? {
                Ok(next) => Some(next),
                Err(e) => break Some(Err(e.into())),
            };
        }
    }

    // There is no way to provide a useful size hint.
}

impl<I> FusedIterator for TryFlatten<I>
where
    I: TryIterator + FusedIterator + Sized,
    I::Ok: TryIterator + FusedIterator,
    <I::Ok as TryIterator>::Error: From<I::Error>,
{
}

/// Iterator for the `TryIteratorExt::try_skip_while` method.
#[derive(Debug, Clone)]
pub struct TrySkipWhile<I, F>
where
    I: TryIterator,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    inner: I,
    predicate: F,
    done_skipping: bool,
}

impl<I, F> TrySkipWhile<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    fn new(inner: I, predicate: F) -> Self {
        TrySkipWhile {
            inner,
            predicate,
            done_skipping: false,
        }
    }
}

impl<I, F> Iterator for TrySkipWhile<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    type Item = Result<I::Ok, I::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = match TryIterator::next(&mut self.inner)? {
                Ok(next) => next,
                Err(e) => break Some(Err(e)),
            };

            if self.done_skipping {
                break Some(Ok(next));
            }

            match (self.predicate)(&next) {
                Ok(true) => (),
                Ok(false) => {
                    self.done_skipping = true;
                    break Some(Ok(next));
                }
                Err(e) => break Some(Err(e)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added, so pass through the upper bound.
        (0, self.inner.size_hint().1)
    }
}

impl<I, F> FusedIterator for TrySkipWhile<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
}

/// Iterator for the `TryIteratorExt::try_take_while` method.
#[derive(Debug, Clone)]
pub struct TryTakeWhile<I, F>
where
    I: TryIterator,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    inner: I,
    predicate: F,
    done_taking: bool,
}

impl<I, F> TryTakeWhile<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    fn new(inner: I, predicate: F) -> Self {
        TryTakeWhile {
            inner,
            predicate,
            done_taking: false,
        }
    }
}

impl<I, F> Iterator for TryTakeWhile<I, F>
where
    I: TryIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
    type Item = Result<I::Ok, I::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done_taking {
            return None;
        }

        let next = match TryIterator::next(&mut self.inner)? {
            Ok(next) => next,
            Err(e) => return Some(Err(e)),
        };

        match (self.predicate)(&next) {
            Ok(true) => Some(Ok(next)),
            Ok(false) => {
                self.done_taking = true;
                None
            }
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Elements cannot be added, so pass through the upper bound.
        (0, self.inner.size_hint().1)
    }
}

impl<I, F> FusedIterator for TryTakeWhile<I, F>
where
    I: TryIterator + FusedIterator + Sized,
    F: FnMut(&I::Ok) -> Result<bool, I::Error>,
{
}

impl<I: ?Sized + TryIterator> TryIteratorExt for I {}
impl<I: ?Sized + DoubleEndedTryIterator> DoubleEndedTryIteratorExt for I {}

/// Import this module to get access to the "normal" set of tryiterator
/// features.
pub mod prelude {
    pub use crate::DoubleEndedTryIteratorExt;
    pub use crate::TryIteratorExt;
    pub use crate::TryPeekableExt;
}
