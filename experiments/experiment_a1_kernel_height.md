# Experiment A1: Best kernel height

Sub-experiment of [experiment_a.md](experiment_a.md). Same data, protocol and metrics.

**Question:** Which kernel height `h` works best for `SpatialNet(spatial="height", kernel_height=h)`?

**Candidates:** `h` ∈ {1, 4, 8, 16, 21}. `h = 21` covers all electrodes at once, as in the original AsymSETNet.

## Open questions (to discuss)

- **How to pick `h`:** separately for each window length, or once (e.g. at 3 s) and reused for every window?
- **Selection criterion:** mean F1 across patients, pooled F1, or something else?
- **Leakage:** picking `h` on the same test folds used in the final comparison makes Experiment A optimistic. Accept that, or select on validation folds only?
- **Channel order:** with `h < 21` the kernel slides over the `REQUIRED_CHANNELS` order, so neighbouring rows are not always neighbouring electrodes. Is that acceptable, or should channels be reordered for this model?
