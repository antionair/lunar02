import ctypes
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROW_BYTES = 4096
ROWS = 65536
FILE_SIZE = ROW_BYTES * ROWS
RANDOM_READS = 10000
TOKENS = 1000
TOKEN_READS = 32
WARMUP_READS = 1000

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
METHOD_NEITHER = 0x3
FILE_ANY_ACCESS = 0
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 4


def make_file(path):
    rng = random.Random(0)
    chunk = rng.randbytes(8 * 1024 * 1024)
    with path.open("wb") as f:
        remaining = FILE_SIZE
        while remaining:
            size = min(len(chunk), remaining)
            f.write(chunk[:size])
            remaining -= size


def aligned_buffer(size):
    VirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
    VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    VirtualAlloc.restype = ctypes.c_void_p
    address = VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not address:
        raise ctypes.WinError()
    return ctypes.c_void_p(address)


def open_direct(path):
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
                            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    CreateFileW.restype = ctypes.c_void_p
    handle = CreateFileW(
        str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    return ctypes.c_void_p(handle)


def random_read(handle, buffer, offset):
    SetFilePointerEx = ctypes.windll.kernel32.SetFilePointerEx
    SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_ulong]
    ReadFile = ctypes.windll.kernel32.ReadFile
    ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                         ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    position = ctypes.c_longlong(offset)
    if not SetFilePointerEx(handle, position, None, 0):
        raise ctypes.WinError()
    count = ctypes.c_ulong(0)
    if not ReadFile(handle, buffer, ROW_BYTES, ctypes.byref(count), None):
        raise ctypes.WinError()
    if count.value != ROW_BYTES:
        raise RuntimeError(f"short read: {count.value}")


def percentiles(samples, values=(0.5, 0.9, 0.99)):
    ordered = sorted(samples)
    result = {}
    for value in values:
        result[f"p{int(value * 100)}"] = ordered[min(len(ordered) - 1, int(value * len(ordered)))]
    return result


def main():
    path = Path(__file__).resolve().parent / "ssd_random_read_test.bin"
    if not path.exists() or path.stat().st_size != FILE_SIZE:
        make_file(path)
    buffer = aligned_buffer(ROW_BYTES)
    handle = open_direct(path)
    rng = random.Random(123)
    try:
        for _ in range(WARMUP_READS):
            random_read(handle, buffer, rng.randrange(ROWS) * ROW_BYTES)
        latencies = []
        started = time.perf_counter_ns()
        for _ in range(RANDOM_READS):
            offset = rng.randrange(ROWS) * ROW_BYTES
            read_started = time.perf_counter_ns()
            random_read(handle, buffer, offset)
            latencies.append((time.perf_counter_ns() - read_started) / 1000)
        random_elapsed = time.perf_counter_ns() - started
        token_latencies = []
        token_started = time.perf_counter_ns()
        for _ in range(TOKENS):
            token_read_started = time.perf_counter_ns()
            for _ in range(TOKEN_READS):
                random_read(handle, buffer, rng.randrange(ROWS) * ROW_BYTES)
            token_latencies.append((time.perf_counter_ns() - token_read_started) / 1000)
        token_elapsed = time.perf_counter_ns() - token_started
        result = {
            "device": "C:",
            "file_bytes": FILE_SIZE,
            "row_bytes": ROW_BYTES,
            "rows": ROWS,
            "random_reads": RANDOM_READS,
            "warmup_reads": WARMUP_READS,
            "latency_us": {
                "mean": statistics.mean(latencies),
                **percentiles(latencies),
                "max": max(latencies),
            },
            "random_read_iops": RANDOM_READS / (random_elapsed / 1e9),
            "random_read_mbps": RANDOM_READS * ROW_BYTES / (random_elapsed / 1e9) / 1e6,
            "token_model": {"token_reads": TOKEN_READS, "tokens": TOKENS},
            "token_latency_us": {
                "mean": statistics.mean(token_latencies),
                **percentiles(token_latencies),
                "max": max(token_latencies),
            },
            "tokens_per_second": TOKENS / (token_elapsed / 1e9),
            "token_read_mbps": TOKENS * TOKEN_READS * ROW_BYTES / (token_elapsed / 1e9) / 1e6,
        }
        print(json.dumps(result, indent=2))
        output = Path(__file__).resolve().parent / "random_read_result.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
        ctypes.windll.kernel32.VirtualFree(buffer, 0, MEM_RELEASE)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
