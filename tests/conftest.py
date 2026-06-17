# 限制 OpenBLAS / OpenMP / MKL 线程数，避免在超多核机器上因线程元数据耗尽而崩溃。
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
