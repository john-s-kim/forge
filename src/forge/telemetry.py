from prometheus_client import Counter, Gauge, Histogram

GPUS_TOTAL = Gauge("forge_gpus_total", "GPUs in the cluster")
GPUS_ALLOCATED = Gauge("forge_gpus_allocated", "GPUs held by running attempts", ["team"])
GPUS_BORROWED = Gauge("forge_gpus_borrowed", "GPUs held beyond the team's guarantee", ["team"])
QUEUE_DEPTH = Gauge("forge_queue_depth", "Jobs waiting", ["priority"])
QUEUE_WAIT = Histogram(
    "forge_queue_wait_seconds",
    "Time from submission to first start",
    ["team", "priority"],
    buckets=(1, 5, 15, 60, 300, 900, 3600, 4 * 3600, 12 * 3600, 86400),
)
CYCLE = Histogram(
    "forge_scheduling_cycle_seconds",
    "Duration of one reconcile cycle",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
)
TRANSITIONS = Counter("forge_job_state_transitions_total", "Job state changes", ["to"])
PREEMPTIONS = Counter("forge_preemptions_total", "Attempts preempted")
STALLED = Gauge("forge_stalled_runs", "Running attempts with no heartbeat for 5 minutes")
IDLE_GPUS = Gauge(
    "forge_idle_allocated_gpus", "GPUs held by attempts that logged no step for 10 minutes"
)
DIVERGED = Counter(
    "forge_diverged_points_total", "Metric points that were NaN or infinite", ["team"]
)
