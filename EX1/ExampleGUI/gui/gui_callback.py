import threading as t

class GuiCallback:

    def __init__(self, locks: tuple[t.Lock], callback: callable, *args):
        self.locks: tuple[t.Lock] = locks
        self.callback = callback
        self.args = args

    def __call__(self, *callback_args):
        aquired_locks = []
        try:
            for lock in self.locks:
                acquired_lock = lock.acquire(blocking=False)

                if not acquired_lock:
                    return

                aquired_locks.append(lock)

            self.callback(*callback_args, *self.args)
        finally:
            for lock in aquired_locks:
                lock.release()
