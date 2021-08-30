"""Python concurrent.futures executors
======================================

This file contains a lazy ThreadPoolExecutor. The ThreadPoolExecutor in Python
standard library first fetches the complete iterable, before using a thread
pool to apply the transformation. This is a major problem for us, as we must
load all data to memory but need to iterate lazily.

"""

import collections
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class LazyThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with lazy iterable collection in map().

    References
    - https://bugs.python.org/issue29842
    - https://github.com/python/cpython/pull/707

    """
    def map(self, fn, *iterables, timeout=None, prefetch=None):
        # pylint: disable=arguments-differ
        """Lazy apdaption of ThreadPoolExecutor.map.

        Unlike ThreadPoolExecutor.map:
        - iterables are prefetched lazily

        """
        if timeout is not None:
            end_time = timeout + time.time()
        if prefetch is None:
            prefetch = self._max_workers
        if prefetch < 0:
            raise ValueError('prefetch count may not be negative')

        argsiter = zip(*iterables)
        initialargs = itertools.islice(argsiter, self._max_workers + prefetch)
        fs = collections.deque(self.submit(fn, *args) for args in initialargs)

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def _result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    if timeout is None:
                        res = [fs[0].result()]
                    else:
                        res = [fs[0].result(end_time - time.monotonic())]

                    # Got a result, future needn't be cancelled
                    del fs[0]

                    # Dispatch next task before yielding to keep pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))

                    yield res.pop()
            finally:
                for future in fs:
                    future.cancel()

        return _result_iterator()


class LazyProcessPoolExecutor(ProcessPoolExecutor):
    """ProcessPoolExecutor with lazy iterable collection in map().

    References
    - https://bugs.python.org/issue29842
    - https://github.com/python/cpython/pull/707

    """
    def map(self, fn, *iterables, timeout=None, prefetch=None):
        # pylint: disable=arguments-differ
        """Lazy apdaption of ProcessPoolExecutor.map.

        Unlike ProcessPoolExecutor.map:
        - iterables are prefetched lazily

        """
        if timeout is not None:
            end_time = timeout + time.time()
        if prefetch is None:
            prefetch = self._max_workers
        if prefetch < 0:
            raise ValueError('prefetch count may not be negative')

        argsiter = zip(*iterables)
        initialargs = itertools.islice(argsiter, self._max_workers + prefetch)
        fs = collections.deque(self.submit(fn, *args) for args in initialargs)

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def _result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    if timeout is None:
                        res = [fs[0].result()]
                    else:
                        res = [fs[0].result(end_time - time.monotonic())]

                    # Got a result, future needn't be cancelled
                    del fs[0]

                    # Dispatch next task before yielding to keep pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))

                    yield res.pop()
            finally:
                for future in fs:
                    future.cancel()

        return _result_iterator()
