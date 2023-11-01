import threading as t
from typing import Callable

class GuiCallback:
    """
    This class is meant a wrapper for Gui classes that handle concurrent user generated events.
    When registering a callback for an event, button or other use this wrapper as a callback.
    By registering on this object the locks that protect the content used by the call 
    a thread-safe performance is ensured.
    
    WARNINGS
    This class will allways acquire all locks in the order they are given in the constructor.
    It is the responsibility of the user to allways register locks in the same order to prevent deadlocks.
    Also, if an event is tries to acquire the lock and it can't it will be terminated.
    This was done because tkinter will block execution until an event is handled.
    
    Private Attributes (use inside class):
    --------------------------------------
    _locks : tuple[threading.Lock]
        A sequence of locks that ensure safety of the callback operation.
    _callback : Callable
        The actual callback to be used.
    _args : tuple
        A tuple of the arguments for the callback.
    
    """

    def __init__(self, locks: tuple[t.Lock], callback: Callable, *args):
        """Creates an instance of a GuiCallback.

        Args:
            locks (tuple[t.Lock]): The sequence of locks that ensure thread-safety.
            callback (Callable): The actual call to use.
            args (tuple): The args for the callback.
        """
        self._locks: tuple[t.Lock] = locks
        self._callback: Callable = callback
        self._args = args

    def __call__(self, *callback_args):
        """
        Perform the thread-safe callback. 
        Important to note that if the lock is acquired the callback will terminate.
        """
        aquired_locks = []
        try:
            for lock in self._locks:
                acquired_lock = lock.acquire(blocking=False)

                if not acquired_lock:
                    return

                aquired_locks.append(lock)

            self._callback(*callback_args, *self._args)
        finally:
            for lock in aquired_locks:
                lock.release()
