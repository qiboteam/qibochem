def limit_gpu_memory(memory_limit=None):
    """Limits GPU memory that is available to Tensorflow.
    Args:
        memory_limit: Memory limit in MBs.
    """
    import tensorflow as tf

    if memory_limit is None:
        print("\nNo GPU memory limiter used.\n")
        return

    print("\nAttempting to limit GPU memory to {}.\n".format(memory_limit))
    for gpu in tf.config.list_physical_devices("GPU"):
        config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)
        tf.config.experimental.set_virtual_device_configuration(gpu, [config])
        print("Limiting memory of {} to {}.".format(gpu.name, memory_limit))
    print()
    return memory_limit


def select_numba_threading(threading):
    from numba import config, threading_layer

    print(f"\nSwitching threading to {threading}.\n")
    config.THREADING_LAYER = threading
    return threading_layer
