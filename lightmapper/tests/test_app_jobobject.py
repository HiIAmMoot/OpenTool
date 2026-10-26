import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


REPO = Path(__file__).resolve().parents[2]
HELPER = r'''
import subprocess, sys
from opentool.jobobject import kill_children_on_exit
enabled = kill_children_on_exit()
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid, enabled, flush=True)
'''
HELPER_WITHOUT_JOB = HELPER.replace('enabled = kill_children_on_exit()', 'enabled = False')


def pid_alive(pid):
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
    finally:
        kernel32.CloseHandle(handle)


def kill_pid(pid):
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.OpenProcess(0x0001, False, pid)
    if handle:
        try:
            kernel32.TerminateProcess(handle, 1)
        finally:
            kernel32.CloseHandle(handle)


def wait_until(predicate, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.mark.skipif(os.name != 'nt', reason='Windows job objects are Windows-only')
def test_grandchild_dies_when_the_app_exits(tmp_path):
    out = subprocess.run([sys.executable, '-c', HELPER], capture_output=True, text=True, cwd=REPO, timeout=60).stdout.split()
    assert out[1] == 'True'
    assert wait_until(lambda: not pid_alive(int(out[0])), timeout=10)


@pytest.mark.skipif(os.name != 'nt', reason='Windows job objects are Windows-only')
def test_without_the_job_the_grandchild_survives(tmp_path):
    pid = int(subprocess.run([sys.executable, '-c', HELPER_WITHOUT_JOB], capture_output=True, text=True, cwd=REPO, timeout=60).stdout.split()[0])
    try:
        assert pid_alive(pid)
    finally:
        kill_pid(pid)
