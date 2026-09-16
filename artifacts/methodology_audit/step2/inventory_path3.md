# Path 3 probability-conversion inventory (evidence for Step 2 §1)

Produced 2026-09-15 by an exhaustive search of scripts/mc_pool_backtest.py, src/optimization/,
src/evaluation/, src/simulation/, src/prediction/, src/data/features/proprietary_metrics.py,
src/cli/pool_cmds.py. Legend: BT = mc_pool_backtest runtime; CAB = candidate artifact builder;
REF = referee/simulation; UI = CLI/UI payload; AUD = referee_audit; T = tests-only; DEAD = no runtime caller.

## Live rows (formula quoted from code)

| file:line | function | in -> out | formula | pw/marg | live in |
|---|---|---|---|---|---|
| src/prediction/pairwise.py:101 | log5 | two barthag -> P(A>B) | num=pa*(1-pb); denom=pa*(1-pb)+pb*(1-pa); denom<1e-12 -> 0.5 | pairwise | BT, CAB, UI, AUD |
| pairwise.py:157 | from_ratings | ratings -> ordered-pair table | p=log5(..); probs[(t2,t1)]=1-p | pairwise | BT, CAB, UI |
| pairwise.py:275 | simulate_bracket_outcomes | pairwise -> per-sim winners | `if rng.random() < p` per game; optional logit noise | pw->outcome | CAB |
| pairwise.py:354 | marginals_from_pairwise | pairwise -> round_probs | MC count, floor 0.001 | pw->marg | BT (pit), CAB |
| mc_pool_backtest.py:852 | build_torvik_round_probabilities | barthag -> marginals | log5 per game, MC, floor 0.001, seed 42 | pw->marg | BT, UI |
| mc_pool_backtest.py:790 | build_pit_base | year -> ProbabilityBase | pairwise_for_year + marginals_from_pairwise | both | BT, AUD |
| mc_pool_backtest.py:2369 | draw_selection_trials | seed_pw, pick_dist -> CRN trials | generate_opponent_brackets + simulate_tournament_outcomes(noise 0.16) | referee | BT, CAB, AUD |
| seed_probabilities.py:46 | build_seed_probabilities | seeds -> pairwise | _win_rate(.., "recent", as_of) | pairwise | BT (referee seed_pw), CAB, AUD, UI |
| seed_probabilities.py:68 | build_seed_round_probabilities | seeds -> marginals | _compute_advancement_rates("recent", as_of) | marg (analytic) | BT, CAB, AUD, UI |
| seed_pick_model.py:211 | _win_rate | seeds -> P | shrunk cell rate or logistic 1/(1+exp(-0.175*(sb-sa))) | pairwise | BT |
| seed_pick_model.py:251 | _compute_advancement_rates | seed pw -> P(seed reaches R) | exact opponent-mixture recursion | pw->marg | BT, UI |
| noseed_model.py:317 | predict_win_prob | stats -> P | 0.5*p_lr + 0.5*sigmoid(spread/11) | pairwise | BT, UI, AUD |
| noseed_model.py:415 | build_noseed_round_probabilities | seed marg x (1+mean_adv)^(i+1), clip [0.001,0.99] | heuristic marginal | BT, CAB, UI, AUD |
| noseed_model.py:455/467 | build_blend_(round_)probabilities | alpha*seed + (1-alpha)*noseed | pw / marg | BT, UI, AUD |
| pit_production_model.py:254 | pairwise_for_year | ridge margin -> clip(student_t(a*m/sigma, nu)); causal shrunk calibration | pairwise | BT, AUD |
| pool_competition.py:428 | simulate_tournament_outcomes | referee pw -> outcomes | clip[.001,.999]; logit+N(0,noise); sigmoid; clip[.01,.99]; Bernoulli | pw->outcome | BT, REF, CAB, UI, AUD |
| pool_competition.py:232/382 | generate_opponent_brackets / _get_pick_prob | ESPN shares -> P(pick t1) = t1/(t1+t2) | marg->pick | BT, REF, CAB, UI, AUD |
| market_probabilities.py:33/313/510 | market rating loaders | implied probs -> barthag (Bradley-Terry / logit avg) | rating | BT, AUD |
| elo/massey/ap/knn/stacked *_probabilities.py | loaders | -> barthag-equivalent, clipped | rating | BT (research bases) |
| meta_selector.py:69 | _pairwise_prob | marginals -> p1/(p1+p2) | marg->pseudo-pw (allow-listed; meta_* GBM modes only) | BT research |
| bracket_construction.py:174 | _make_ev_scorer | marginal*pts*uniqueness weight | marg | BT, CAB, UI |
| conditional_bracket_engine.py:168 | expected_scores | sum pts_R * P(pick wins R) | marg | CAB |
| referee_audit.py:313 | load_fte_pairwise | norm.cdf((r_a - r_b)/11) | pairwise | AUD |

Dead (no runtime caller): src/simulation/monte_carlo.py (MonteCarloEngine, _run_batch), pool_competition.run_pool_simulation / PoolCompetitionSimulator, dual_submission.py, competitor_archetypes.py, matchup_vulnerability.py, mc_pool_backtest.build_optimized_brackets (+ _compute_game_confidence, _generate_bracket_variants), torvik_kaggle.EnsembleKagglePredictor, bracket_portfolio (tests only), pool_optimizer._build_round_probabilities (unreachable fallback).

## Q1 referee and noise
seed_pw = build_seed_probabilities(seeds[, as_of=year]) at mc_pool_backtest.py:2861; passed as matchup_probs to
draw_selection_trials (:3710, :4027, :4226) and to the final referee (:4556-4567, :4615-4626).
REFEREE_NOISE_STD = 0.16 (:203) -> run_backtest -> _run_one_year -> simulate_tournament_outcomes, applied at
pool_competition.py:474-483: safe_p=clip(p,.001,.999); logit+=N(0,noise); final=sigmoid; clip(.01,.99); Bernoulli.
Opponent field uses a separate chalk_noise_std (pool-level logit shift), default 0.0 in the canonical run.

## Q2 simulators
Live: pool_competition.simulate_tournament_outcomes (referee) and pairwise.simulate_bracket_outcomes /
marginals_from_pairwise (candidate bank, pit base). monte_carlo.py is dead. Analytic marginals exist only for the
seed table (_compute_advancement_rates); all others are MC counts.

## Q3 clips
See STEP2 §10; full list retained in the agent transcript. Load-bearing: referee [.001,.999]/[.01,.99];
marginals floor 0.001; PROB_CLIP 1e-3 (pit/fit.js); barthag fallback max(0.10, 1-seed*0.04); Massey [0.10,0.99].

## Q4 marginal -> pairwise sites
meta_selector._pairwise_prob (meta_* GBM modes, allow-listed); _bracket_export_common.build_bracket_json (dead
export docs/data/bracket_2026.json, allow-listed); bracket_construction._get_prob (SA mode label only);
pool_competition._get_pick_prob (opponent pick model, not outcome); pool_optimizer._build_round_probabilities
(reverse direction, unreachable). None on the meta_region_poolaware or shipped-artifact path.

## Q5 log5 implementations
Canonical pairwise.log5; delegating alias mc_pool_backtest._log5; independent copies torvik_kaggle._log5,
meta_selector._log5 (nested), loyo_pergame_predictions._log5 (script is broken: imports a module deleted in
ea06a40); massey_best._pairwise_win_prob is algebraically log5. All compared to canonical: max diff 0.
proprietary_metrics._log5_win_prob is NOT log5 (logistic on efficiency margin, k=11.5).
