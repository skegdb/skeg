"""Parse kernel counters, not only Docker configuration or shell exit status."""
import re


def validate(text, limit_mib):
    maximum = re.search(r"^memory.max:\n(\d+)\s*$", text, re.MULTILINE)
    peak = re.search(r"^memory.peak:\n(\d+)\s*$", text, re.MULTILINE)
    assert maximum and peak, "missing kernel memory limit/peak"
    assert int(maximum[1]) == limit_mib * 1024 * 1024, "wrong kernel cgroup limit"
    assert 0 < int(peak[1]) <= int(maximum[1]), "invalid cgroup peak"
    assert "memory.events:\n" in text, "missing memory events"
    events = dict((key, int(value)) for key, value in re.findall(
        r"^(\w+) (\d+)$", text.split("memory.events:\n", 1)[1], re.MULTILINE))
    # The test child could be OOM-killed while PID 1 (the shell) survives.
    assert events.get("oom") == 0 and events.get("oom_kill") == 0, "cgroup OOM event"
    assert events.get("oom_group_kill", 0) == 0, "cgroup group OOM event"
    return dict(memory_max_bytes=int(maximum[1]), memory_peak_bytes=int(peak[1]), memory_events=events)
