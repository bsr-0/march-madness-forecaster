# Iteration Summary: Construction-First → Opponent-Aware (2026-05-01/02)

## Where We Are

**`meta_region_poolaware` achieves 11.9% P(1st)** — 4x the seed baseline (3.1%), validated at p=0.0076 after multiple comparison correction across 20+ strategies on 14 LOYO years.

The system generates ~25 diverse candidate brackets per year (varying construction mode, probability base, risk level, and forced champion), then simulates each against the opponent field to select the one most likely to finish first. The key insight: optimizing P(beat field) instead of E[points] is a fundamentally different axis that broke through the 8% ceiling.

## What We Tried and Killed

- GBM learned model (4.6%) — worse than raw probability-based construction
- GBM probabilities fed into construction modes — less calibrated than torvik, hurt P(1st)
- SA construction — fundamentally broken (1-2%)
- 14 Kaggle/academic techniques — none beat region construction
- Champion pick optimization — irrelevant to P(1st) (proven by experiment)
- Upset specialist classifier — null result (7.9%, can't beat beam search on coin-flip games)
- Volatility-adapted risk signal — regressed GBM, killed

## Plan Forward

1. **Lock `meta_region_poolaware` as 2027 production strategy.** The expanded candidate pool (5 risk levels × 5 probability bases × 2 construction modes + 4 forced champions, deduped to ~25 unique brackets) with opponent-aware selection is the strongest approach found.

2. **Expand opponent model.** Currently uses generic opponent generation. Incorporating the 105 real pool brackets (2023-2026) as a behavioral prior for opponent simulation could further improve selection accuracy.

3. **More candidate diversity.** The `blend` probability base was selected 7/14 years — exploring more blend ratios and additional probability sources (e.g., walk-forward Massey best-system selection) could expand the useful candidate frontier.

4. **Accept diminishing returns on prediction.** The council confirmed: probability improvement is dead (BSS=0 ceiling). All future gains come from construction innovation and opponent modeling, not better game prediction.
