"""Attribution run: today's code, but the noseed marginals on the OLD (full, no as_of) window,
so the difference between this run and the full-fix run isolates F3-4."""
import sys, runpy
sys.path.insert(0, '/Users/benrosen/Documents/march-madness-forecaster')
import src.prediction.noseed_model as NM
_orig = NM.build_noseed_round_probabilities
def _old_window(model, seeds, stats, *, as_of, window="recent"):
    return _orig(model, seeds, stats, as_of=None, window="full")
NM.build_noseed_round_probabilities = _old_window
import scripts.mc_pool_backtest as M
M.build_noseed_round_probabilities = _old_window
sys.argv = ["mc_pool_backtest", "--team-identity", "--opponent", "pool", "--n-opponents", "29", "--n-repeats", "100", "--modes", "seed", "meta_region_poolaware", "--no-log"]
M.main()
