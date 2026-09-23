"""Multi-core parallel computing (multiprocessing) for ANN training and architecture search.

A single process pool is shared and closes itself after being idle for a while, so it does not use memory when
not needed. Each job is a module-level function `fn(*args, cancel=...)`; in the worker processes `cancel()` reads
a shared event so the Stop button keeps working.

With workers <= 1 all jobs run sequentially in this process (results identical to the parallel version).
"""
import atexit
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

IDLE_SECONDS = 90          # the pool is closed after being idle this long

_lock = threading.RLock()
_pool = None
_pool_n = 0
_event = None
_users = 0
_last = 0.0
_timer = None
_worker_event = None       # in the worker processes


def cpu_count():
    return os.cpu_count() or 1


def default_workers():
    """All logical cores minus one (so the computer stays responsive)."""
    return max(1, cpu_count() - 1)


def is_warm():
    """True when the process pool is already running (no cost of starting new processes)."""
    with _lock:
        return _pool is not None and not getattr(_pool, "_broken", False)


def _init_worker(event):
    global _worker_event
    _worker_event = event
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[var] = "1"


def _worker_cancelled():
    return _worker_event is not None and _worker_event.is_set()


def _call(fn, args):
    return fn(*args, cancel=_worker_cancelled)


def _acquire(n):
    global _pool, _pool_n, _event, _users
    with _lock:
        if _pool is not None and (getattr(_pool, "_broken", False) or (_pool_n != n and _users == 0)):
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        if _pool is None:
            # child processes inherit the environment when created; BLAS must be single-threaded before numpy
            # is loaded in the child
            for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            ctx = mp.get_context("spawn")
            _event = ctx.Event()
            _pool = ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                        initargs=(_event,))
            _pool_n = n
        _users += 1
        return _pool, _event


def _release(broken=False):
    global _users, _last, _timer, _pool
    with _lock:
        _users = max(0, _users - 1)
        _last = time.monotonic()
        if broken and _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        if _users == 0 and _pool is not None:
            if _timer is not None:
                _timer.cancel()
            _timer = threading.Timer(IDLE_SECONDS + 1, _close_if_idle)
            _timer.daemon = True
            _timer.start()


def _close_if_idle():
    global _pool
    with _lock:
        if _pool is not None and _users == 0 and time.monotonic() - _last >= IDLE_SECONDS:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def shutdown():
    global _pool
    with _lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


atexit.register(shutdown)


def run_jobs(fn, jobs, workers=1, cancel=None, on_result=None):
    """Run fn(*job, cancel=...) for each job; results are returned in job order.

    on_result(i, result) is called whenever a job finishes (in any order). When cancelled, jobs that have not
    started are skipped (result None) and running jobs are asked to stop through cancel() in the worker process.
    """
    jobs = list(jobs)
    out = [None] * len(jobs)
    n = min(int(workers or 1), len(jobs))
    if n <= 1:
        for i, args in enumerate(jobs):
            if cancel is not None and cancel() and i > 0:
                break
            out[i] = fn(*args, cancel=cancel)
            if on_result:
                on_result(i, out[i])
        return out
    pool, event = _acquire(int(workers))
    broken = False
    pending = set()
    try:
        if cancel is None or not cancel():
            event.clear()
        futs = {pool.submit(_call, fn, args): i for i, args in enumerate(jobs)}
        pending = set(futs)
        stop_sent = False
        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for f in done:
                if f.cancelled():
                    continue
                i = futs[f]
                out[i] = f.result()           # MemoryError etc. are passed on to the caller
                if on_result:
                    on_result(i, out[i])
            if cancel is not None and not stop_sent and cancel():
                stop_sent = True
                event.set()
                for f in pending:
                    f.cancel()
        return out
    except BrokenProcessPool:
        broken = True
        raise MemoryError("A parallel process stopped unexpectedly (possibly not enough memory).") from None
    finally:
        for f in pending:
            f.cancel()
        _release(broken)
