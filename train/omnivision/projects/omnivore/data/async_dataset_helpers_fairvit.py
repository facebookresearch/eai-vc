# Exact copy of https://github.com/fairinternal/fairvit/blob/master/app/dino/async_dataset.py
# TODO: fairvit installation doesn't install things under the `app` folder
# If the installation is changed, we can directly import the file from it

import asyncio
import collections
import concurrent
import contextlib
import itertools
import os
import sys
import threading
import weakref
from io import BytesIO

from typing import Any, Awaitable, Dict, Generic, Iterable, Iterator, TypeVar

from PIL import Image

from torch.utils import data
from torch.utils.data.sampler import Sampler


class ChannelClosedError(ValueError):
    ...


T = TypeVar("T")


class AsyncToSyncChannel(Generic[T]):
    """
    Bounded MPMC channel with async writer and sync reader.

    Why this is needed
    ------------------

    `queue.Queue` and `queue.SimpleQueue` provide queues between threads.
    `asyncio.Queue` provides queues between async tasks.

    However, there is no bounded queue type for sending data between async tasks
    and threads. This class provides that for the async->sync direction.

    Cleanup
    -------

    Both the reader and writer should wrap the queue in a 'with' block::

        async def writer(queue):
            with queue:
                for x in range(100): queue.put(x)

                                                                                            def reader(queue):
            with queue:
                for x in queue: print(x)

    This will call `close()` when the block exits, which ensures that the other
    end will not hang forever waiting for an item that won't arrive.

    If the reader exits first, the writer will fail with `ChannelClosedError`
    when it tries to put to a closed queue. If the writer exits first, the
    reader will first process the remaining items normally, then get
    `StopIteration` (i.e. simply exit the for-loop normally).
    """

    def __init__(self, maxsize):
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._nonempty_event = threading.Event()
        self._closed = False
        self._maxsize = maxsize
        self._put_waiter = None

    async def put(self, obj: T):
        """
        Put an item onto the queue.

        Blocks if the queue is full. Raises ValueError if the queue is closed.
        """
        while True:
            with self._lock:
                if self._closed:
                    raise ChannelClosedError("attempt to push to closed queue")
                if len(self._queue) < self._maxsize:
                    self._queue.append(obj)
                    self._nonempty_event.set()
                    return True
                else:
                    f = self._put_waiter
                    if f is None:
                        self._put_waiter = f = asyncio.Future()
            await f

    def get(self) -> T:
        while True:
            with self._lock:
                if len(self._queue) > 0:
                    # wake up writer since we're making some room
                    self._awaken_writers()
                    return self._queue.popleft()
                elif self._closed:
                    raise StopIteration
                else:
                    self._nonempty_event.clear()
            self._nonempty_event.wait()

    def close(self, discard=False):
        """
        Mark the queue as closed.

        Attempting to `put` to a closed queue raises ValueError.

        Attempting to iterate a closed queue will continue to yield any remaining
        items (unless discard=True) and then raise `StopIteration`.

        Parameters
        ----------
        discard : bool
            If True, any pending items in the queue will be discarded.
        """
        with self._lock:
            self._closed = True
            if discard:
                self._queue.clear()
            self._nonempty_event.set()
            self._awaken_writers()

    def closed(self) -> bool:
        return self._closed

    # implement iterable interface for consuming queue
    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return self.get()

    # implement context manager interface for closing queue
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _awaken_writers(self):
        # awaken async task waiting for _put_waiter.
        # this can only be called when _lock is held!
        assert self._lock.locked()
        if self._put_waiter:
            self._put_waiter.get_loop().call_soon_threadsafe(
                self._put_waiter.set_result, None
            )
            self._put_waiter = None


def _shard_iterator_dataloader_worker(iterable):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        yield from itertools.islice(
            iterable, worker_info.id, None, worker_info.num_workers
        )


_async_thread = None


def _async_main(loop):
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def get_event_loop():
    global _async_thread
    th = _async_thread
    if th is not None and th.is_alive():
        return th.loop

    if th is None:
        pass
        # print(f"creating event loop thread for {os.getpid()}")
    elif th.original_pid == os.getpid():
        # thread died somehow? don't recreate it.
        raise RuntimeError("event loop thread is dead")
    else:
        pass
        # thread not running because we're in a forked process
        # print(
        #    f"recreating event loop thread for {os.getpid()} after fork from {th.original_pid}"
        # )

    loop = asyncio.new_event_loop()
    th = threading.Thread(
        target=_async_main, name="s2_async", args=(loop,), daemon=True
    )
    th.original_pid = os.getpid()
    th.loop = loop

    _async_thread = th
    th.start()
    return loop


class _AFuture(concurrent.futures.Future):
    """
    (Internal) Future type with cleaner cancel semantics.

    Allows for tasks to actually cancel when a future is cancelled.  This is as
    opposed to immediately reporting the task done without any possibility of
    cleanup.
    """

    def __init__(self, coro_or_task, loop):
        if not (
            asyncio.iscoroutine(coro_or_task) or isinstance(coro_or_task, asyncio.Task)
        ):
            raise TypeError("A coroutine or task object is required")

        super().__init__()
        self._loop = loop
        self._task = None
        self._cancelled = False

        if isinstance(coro_or_task, asyncio.Task):
            self._task = coro_or_task
            coro_or_task.add_done_callback(self._set_state)
        else:
            loop.call_soon_threadsafe(self._create_task_cb, coro_or_task)

    def cancel(self):
        # Cancel the async task associated with this future.
        # This does *not* call super.cancel(), because
        # concurrent.futures.Future makes it impossible to wait for
        # a result once that has been done. Instead it schedules
        # an async cancellation and then lets that propagate normally.
        self._loop.call_soon_threadsafe(self._cancel_cb)
        return True

    def _create_task_cb(self, coro):
        loop = self._loop
        self._task = task = loop.create_task(coro)
        task.add_done_callback(self._set_state)
        if self._cancelled:
            task.cancel()

    def _cancel_cb(self):
        if not self._cancelled:
            self._cancelled = True
            if self._task:
                self._task.cancel()

    def _set_state(self, task):
        assert task.done()
        if task.cancelled():
            super().cancel()
        if not self.set_running_or_notify_cancel():
            return
        exc = task.exception()
        if exc is not None:
            # convert to concurrent.futures.CancelledError to match concurrent
            # API. We don't have to convert TimeoutError because it will
            # already be the concurrent.future version.
            if isinstance(exc, asyncio.CancelledError):
                exc = type(exc)(*exc.args)
            self.set_exception(exc)
        else:
            self.set_result(task.result())


def run_coroutine_threadsafe(
    coro_or_task: Awaitable[T], loop: asyncio.AbstractEventLoop
) -> concurrent.futures.Future:
    """
    Like :func:`asyncio.run_coroutine_threadsafe` but cancellation works properly.

    The problem:

    :meth:`concurrent.futures.Future.cancel` sets the future to a "done" state
    *immediately*. It provides no way whatsoever to wait for cleanup to complete,
    nor to report errors that occurred during cleanup.

    :meth:`asyncio.run_coroutine_threadsafe` returns a Future so it inherits this
    problem. This leads to race conditions in cleanup code and makes it nearly
    impossible to robustly shut things down.

    This function returns a derived `Future` that overrides `cancel()` to request
    the task be cancelled, but doesn't mark the Future as done until the task has
    actually finished cancellation.

    In other words, if you do::

        state = "initial"
        async def my_async_fn():
            global did_cleanup
            try:
                state = "dirty"
                await asyncio.sleep(100)
            finally:
                state = "clean"

        f = run_coroutine_threadsafe(my_async_fn(), loop)
        ...
        f.cancel()
        concurrent.futures.wait([f])
        assert state in ("initial", "clean")

    If the coroutine made it into the 'try' block, then the `wait()` call
    (or fetching `result()` or `exception()`, which implicitly wait) will
    not return until the "finally" block has run too.

    Unlike the asyncio version, you can also pass an existing `Task` object
    instead of a coroutine, so that callers can control task creation if
    necessary.

    See also
    --------
    :meth:`spawn`, which just calls this function using the
    default nested_async loop.
    """
    return _AFuture(coro_or_task, loop)


def spawn(awaitable):
    return run_coroutine_threadsafe(awaitable, get_event_loop())


async def wait(fs, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
    """Same as asyncio.wait but allows empty list of futures."""
    if fs:
        return await asyncio.wait(fs, timeout=timeout, return_when=return_when)
    else:
        return (set(), set())


class TaskGroup:
    """
    An async context manager that waits for child tasks on exit.

    When writing parallel async code (i.e. using create_task to actually run
    tasks in parallel), it can be difficult to ensure everything is cleaned up
    if one task fails with an error.

    This context manager keeps track of a set of tasks that should all be
    complete by the time the group context exits. If the context exits normally,
    it waits for all the tasks to complete. If it exits via an exception, it
    cancels all tasks and then waits for them.

    Tasks can be added to the context manager explicitly via :meth:`add`, or
    (more easily) by using :meth:`create_task` in place of
    :func:`asyncio.create_task`

    Example::

        async def read_blob(name): ...

        async def concat_blobs(name1, name2):
            async with async_utils.TaskGroup() as tgroup:
                b1 = tgroup.create_task(read_blob(name1))
                b2 = tgroup.create_task(read_blob(name2))
                return await b1 + await b2

    If the context exits via an exception, any still-running tasks in the group
    will be cancelled (unless cancel_on_error=False). Any additional errors
    thrown during cleanup will be discarded.

    On normal exit, it will wait for all tasks to complete. If any of them raise
    an error, the remaining tasks will be cancelled.

    This means there's a "join" at the end of the with block, so the fact that
    `concat_blobs` contains concurrency is irrelevant to its callers, enabling
    compostion of the code.

    There's an exellent explanation of this model at
    https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/
    with :class:`TaskGroup` being equivalent to Trio's `nursery`, and
    `group.create_task()` equivalent to `nursery.start_soon()`.
    """

    def __init__(self, cancel_on_error=True):
        self.tasks = weakref.WeakSet()
        self.cancelled = False
        self.exceptions = []
        self.context_task = None
        self._cancel_on_error = cancel_on_error

    def add(self, task: asyncio.Task) -> asyncio.Task:
        """
        Add an existing Task to this manager.
        """
        if self.cancelled:
            raise ValueError("attempting to add task to cancelled taskgroup")
        self.tasks.add(task)
        return task

    if sys.version_info >= (3, 8):
        # 'name' param added in 3.8
        def create_task(self, awaitable, name=None) -> asyncio.Task:
            """
            Start a task managed by this TaskGroup.
            """
            if self.cancelled:
                raise ValueError("attempting to create task in cancelled taskgroup")
            return self.add(asyncio.create_task(awaitable, name=name))

    else:

        def create_task(self, awaitable, name=None) -> asyncio.Task:
            """
            Start a task managed by this TaskGroup.
            """
            del name
            if self.cancelled:
                raise ValueError("attempting to create task in cancelled taskgroup")
            return self.add(asyncio.create_task(awaitable))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            if self._cancel_on_error or isinstance(exc, asyncio.CancelledError):
                self.cancel()

        # wait for all pending tasks to finish
        cleanup_error = None
        while self.tasks:
            try:
                done, _ = await wait({*self.tasks}, return_when=asyncio.FIRST_EXCEPTION)
                self.tasks -= done
            except asyncio.CancelledError as c:
                # the context task was cancelled before `wait` completed.
                # Child tasks may not be done yet, so we have to wait for
                # them again, but we'll no longer exit with success.
                self.cancel()
                cleanup_error = c

            # save and reraise a non-cancellation exception if we hit one,
            # so we don't lose track of real errors.
            if exc_type is None and cleanup_error is None:
                for t in done:
                    if not t.cancelled() and t.exception():
                        cleanup_error = t.exception()
                        self.cancel()
                        break

        if exc_type is None and cleanup_error:
            raise cleanup_error

    def cancel(self):
        """
        Cancel all tasks in the group.

        `cancel()` must be idempotent to avoid double-cancelling child tasks;
        doing so would raise CancelledError in them twice and interrupt any
        cleanup they may have been doing.
        """
        if not self.cancelled:
            self.cancelled = True
            not_done = {t for t in self.tasks if not t.done()}
            for t in not_done:
                t.cancel()


async def _put_prefetch_tasks(seq, channel):
    def decrement_inflight_count(_):
        channel.count_inflight -= 1

    async with TaskGroup(cancel_on_error=False) as taskgroup:
        # Iterate over the sequence, starting an async task for each element.
        # The bounded channel limits the number of tasks we'll have in flight.
        with channel, contextlib.closing(seq) as it:
            try:
                for awaitable in it:
                    task = taskgroup.create_task(awaitable)
                    channel.count_inflight += 1
                    task.add_done_callback(decrement_inflight_count)
                    await channel.put(run_coroutine_threadsafe(task, task.get_loop()))
            except ChannelClosedError:
                taskgroup.cancel()


def prefetch_sequence(seq: Iterable[Awaitable[T]], prefetch_size: int) -> Iterable[T]:
    """
    Converts an iterable of Awaitable[T] to an iterable of T

    Runs an event loop on a worker thread to process `prefetch_size`
    elements in parallel.

    Note: this takes a regular (non-async) iterable of awaitables, not
          an "async iterable".
    """
    channel = AsyncToSyncChannel(maxsize=prefetch_size)
    channel.count_inflight = 0  # pyre-ignore
    prefetch_fut = spawn(_put_prefetch_tasks(seq, channel))
    try:
        with channel:
            # count = 10
            # idx = 0
            # inflight_counts = []
            for f in channel:
                # if idx < count:
                #    inflight_counts.append(channel.count_inflight)
                #    idx += 1
                # else:
                #    print("inflight", sum(inflight_counts) / count)
                #    idx = 0
                #    inflight_counts = []
                y = f.result()
                yield y
                del y
                del f

    except GeneratorExit:
        # Our caller did a 'break' from our sequence.
        # Discard any pending work. If that produces an error other than
        # cancellation, that error will be re-raised.
        prefetch_fut.cancel()
        with contextlib.suppress(concurrent.futures.CancelledError):
            prefetch_fut.result()
        raise
    except BaseException:
        # discard errors during cancellation in favor of the
        # one that got us here.
        prefetch_fut.cancel()
        concurrent.futures.wait([prefetch_fut])
        raise
    prefetch_fut.result()


class AsyncToIterableDataset(data.IterableDataset):
    """
    Turn an async dataset to an iterable dataset. The iterable will
    have the order determined by the given sampler.
    """

    def __init__(
        self, dataset: data.Dataset, sampler: Sampler, max_prefetch: int = 256
    ):
        """
        Args:
            dataset: a map-style dataset with async __getitem__.
            sampler:
            max_prefetch: number of elements to prefetch. This also determines how many
                async requests to launch in parallel.
        """
        self.dataset = dataset
        self.sampler = sampler
        self.max_prefetch = max_prefetch

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        sampler = _shard_iterator_dataloader_worker(self.sampler)
        samples = (self.dataset[i] for i in sampler)
        with contextlib.closing(prefetch_sequence(samples, self.max_prefetch)) as seq:
            yield from seq


# Alias for dataset of awaitables
AsyncDataset = data.Dataset[Awaitable[T]]


async def _get_image_async(fpath: str, transform=None):
    img = None
    with open(fpath, "rb") as hFile:
        img = Image.open(hFile)
        img = img.convert("RGB")
    if img is not None and transform is not None:
        img = transform(img)
    return img


class AsyncDatasetWrapper(AsyncDataset[Dict[str, Any]]):
    def __init__(self, dataset: data.Dataset):
        """
        Args:
            dataset: a map-style (indexable) dataset. It must be cheap to index.
                For example it can be a `DatasetFromList`.
                Each element must be a Dict[str, Any].
            read_keys: mapping from the dictionary key that contains manifold path
                to the new dictionary key to save read manifold data, e.g. {"file_name": "image"}.
                The output dictionary will contain a new field "image", with data read from field "file_name".
                The data is represented as raw bytes.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    async def __getitem__(self, idx: int) -> Dict[str, Any]:
        if hasattr(self.dataset, "getitem_async"):
            return self.dataset.getitem_async(idx)
        fpath, target = self.dataset[idx]
        if hasattr(self.dataset, "get_image_fileobj"):
            data = self.dataset.get_image_fileobj(idx)
        elif hasattr(self.dataset, "get_image_path"):
            fpath = self.dataset.get_image_path(idx)
            data = open(fpath, "rb").read()
        else:
            data = open(fpath, "rb").read()
        return data, target


def async_reader(
    dataset: data.Dataset,
    sampler: Sampler,
    max_prefetch: int = 256,
):
    """
    A simple wrapper of AsyncDatasetWrapper + AsyncToIterableDataset

    Args:
        dataset: a map-style (indexable) dataset. It must be cheap to index.
            For example it can be a `DatasetFromList`.
            Each element must be a Dict[str, Any].
        sampler: a sampler
        max_prefetch: number of elements to prefetch. This also determines how many
            async requests to launch in parallel.
    """
    ds = AsyncDatasetWrapper(dataset)
    return AsyncToIterableDataset(ds, sampler, max_prefetch)


class ImageLoadingIterableDataset(data.IterableDataset):
    def __init__(self, dataset: data.Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for fdata, label in self.dataset:
            if isinstance(fdata, bytes):
                fdata = BytesIO(fdata)
            img = Image.open(fdata).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            yield img, label

    def __len__(self):
        return len(self.dataset)

    @property
    def sampler(self) -> int:
        return self.dataset.sampler


class ImageLoadingIndexedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        fpath, label = self.dataset[idx]
        with open(fpath, "rb") as hFile:
            img = Image.open(hFile).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)
