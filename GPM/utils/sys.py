import psutil
import resource


def set_memory_limit(PERCENTAGE_MEMORY_ALLOWED=0.9):
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    soft = int(available_memory * PERCENTAGE_MEMORY_ALLOWED)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f'Soft: {soft / 1024 / 1024 / 1024:.2f}G, Hard: {available_memory / 1024 / 1024 / 1024:.2f}G')
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
