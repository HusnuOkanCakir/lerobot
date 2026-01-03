import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity

def main():
    prof_dir = os.environ.get("PROF_DIR", "/home/okan/lerobot/prof/profile_training.py")
    os.makedirs(prof_dir, exist_ok=True)

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"[Profiler] Saving trace to: {prof_dir}")
    print(f"[Profiler] CUDA available: {torch.cuda.is_available()}")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
    ) as prof:
        lerobot_train_main()

    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    except Exception:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

if __name__ == "__main__":
    main()