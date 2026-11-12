# 0005 · Network degradation is applied by OS tools; the code only labels runs

**Context.** The team uses macOS, Windows and Linux, and Dk asked that the report describe the condition (+200 ms RTT, 5 Mbit/s cap), not a specific tool.

**Decision.** The code takes `--network {normal,degraded}` and stores it on rows and in file names. The README documents one way per OS (macOS Network Link Conditioner or dnctl/pfctl, Linux tc netem, Windows clumsy).

**Alternatives.** Shaping traffic from Python (unreliable, OS-specific, needs root). A proxy that injects delay (changes the measured path).

**Consequences.** The user must remember to toggle degradation. The run metadata records the label, and the analysis groups by it.
