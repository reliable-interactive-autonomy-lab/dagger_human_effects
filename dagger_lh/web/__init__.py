"""Browser-based deployment of the DAgger / learned-helplessness study.

The simulator cannot run in a browser -- robosuite is Python and MuJoCo needs a
real GL context -- so the architecture is a thin client over a server-side
simulation: the browser sends key state and receives JPEG frames, while
robosuite, the policy and the LoRA finetuning all stay on the server.

One OS process per participant. Threads were measured and do not work: MuJoCo's
offscreen GL contexts are not thread-safe, and a threaded test had to be killed.
Processes also isolate robomimic's module-level observation-spec globals and each
participant's torch thread pool. Measured on an M3 Max, 8 concurrent processes
each doing step + render + JPEG encode ran at 15 ms/step against a 50 ms budget.
"""
