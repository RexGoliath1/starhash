import functools
import time

def timing_decorator(func):
    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        t_initial = time.time()
        output = func(*args, **kwargs)
        t_end = time.time()
        print("    [TIME DECORATOR]: the ({}) function completed in {} seconds".format(func.__name__, round((t_end - t_initial),5)))
        return output
    return timing_wrapper
