# PCP paper and environment comparison, 8 October 2026

The new Adam trials did not change PCP's physical transitions, reset distribution, penalties, capture action or termination rule. They use the same simulator bytes as the earlier SoftRole RMSprop trials. A relevant difference from the historical public HetNet path is the corrected **observation** construction: each agent now receives its own copy before blind-agent masking. Both the current SoftRole path and paper-v1 HetNet use this correction. This is part of the observation model of the POMDP, so it would be inaccurate to say that every aspect of the environment is identical to the historical release. It does not imply that the dynamics should now be changed to make scores resemble the publication.

## What the publication actually reports

The [AAMAS paper](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf), p. 1179, Table 1, reports frozen converged policies on 50 evaluation trials. Its HetNet row is:

| Domain | Mean episode steps ± SE | Mean cumulative return ± SE |
|---|---:|---:|
| PP | 8.30 ± 0.25 | -0.232 ± 0.010 |
| PCP | 9.98 ± 0.36 | -0.364 ± 0.017 |
| FC | 46.40 ± 2.90 | -9.862 ± 2.77 |

Figure 3 is a **training** plot over three seeds, with epochs on the horizontal axis. The PCP HetNet curve ends near ten steps. That endpoint is a visual description, not recovered numerical data. The text separately reports 9.90 ± 0.58 PCP steps for Binary with 64-bit messages. Table 1's row is labeled HetNet, without a Real/Binary suffix.

The [authors' supplement](https://github.com/CORE-Robotics-Lab/HetNet/blob/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf), §§1.1–1.2 and 2.1, specifies native PCP as two P agents and one A agent on a 5×5 grid, horizon 80, stationary prey, and -0.05 until the relevant objective is complete. It gives three HetGAT layers, four attention heads, Adam 0.001 and training seeds 0–2. The exact numeric sensing radius, samples behind each publication epoch and publication-producing checkout/panel have not been recovered. Radius 2 comes from the public executable defaults.

Downloaded paper and supplement SHA256 identities exactly match the previously recorded primary evidence. Relevant full PDF pages were rendered and visually inspected; the new copies and rendered pages are in this folder.

## Reward units and what 'steps' means

The [current official postprocessor](https://raw.githubusercontent.com/CORE-Robotics-Lab/HetNet/master/print_plot_eval.py) extends a list with per-episode reward vectors and calls `np.mean(rewards)` without an axis. Its scalar is a mean across both agents and episodes. SoftRole's principal `team_return` sums physical-agent returns first. With a fixed 2P1A team, comparable mean-agent return is therefore `team_return / 3`. This is a reporting conversion; it does not change rewards or improve a policy. Today's reporting script does not establish the exact historical Table 1 pipeline.

In both our training and frozen evaluation, an episode step is one joint environment transition, not the sum of three agents' actions. The public evaluator similarly records `num_steps`. SoftRole uses the same `run_episode` logic for training and frozen evaluation; actions and bits remain sampled. Calling `model.eval()` freezes relevant module behavior but does not switch these policies to greedy actions, and there is no easier evaluation-only dynamics branch. The archived adapter sets simulator `args.eval=False`; explicit seeded scenarios supply evaluation initializations, avoiding the release's hardcoded local reset-file path. Different scenario panels can still produce different realized task difficulty.

A training epoch/window mixes policies changing during training; a frozen evaluation measures one checkpoint on independently seeded scenarios. A minimum epoch value, last epoch, last-window average and Table 1 are consequently different estimands. Epochs also need not represent equal numbers of samples or optimizer updates across learners. Compare our methods at a declared environment-step budget with exact actual overshoot and updates retained, then use a shared frozen panel. Treat the published line as historical context unless the original execution protocol is recovered.

## Source-level physical invariants

[audit.py](audit.py) compares Python ASTs, which ignore whitespace/comments, as well as actual byte hashes. `comparison.json` records every input path and SHA256. For the new Adam simulator, old archived SoftRole simulator, working simulator, current paper-v1 simulator, and freshly retrieved [official upstream simulator](https://raw.githubusercontent.com/CORE-Robotics-Lab/HetNet/master/envs/ic3net_envs/predator_capture_env.py), these methods have identical ASTs:

- `step`, `reset`, `_get_cordinates`, `_take_action`, `_get_reward`, `reward_terminal`.

New Adam and old SoftRole raw PCP simulator files are also byte-identical (`935875ac...9583`). Adam's archive and prior shared RMSprop configurations agree on every existing field. Only two keys were added: the explicit historical observation-version name and `optimizer='adam'`. The previous source already used that observation version implicitly. All three Adam seeds use native 2P1A, radius 2, 5×5, horizon 80, no sensor failures, no variable compositions and unlimited communication.

The preserved mechanics are:

- Distinct initial P, A and target cells are sampled uniformly, without replacement.
- The target stays fixed. Agents can share cells, and do not block each other.
- P reaches the target; A reaches it and must take a separate capture action.
- A's capture test only checks that A is at the target. It does **not** require a P to have discovered or reached the target beforehand.
- A cannot leave once it reaches the target, and P cannot leave after reaching. These absorbing behaviors are present upstream.
- P stops paying -0.05 on the reaching transition; A stops on the capture transition. Success requires all agents to complete. A failed episode ends at the 80-step horizon. There is no added terminal bonus in these runs.

These code comparisons establish continuity of the executed local PCP physics and the current public implementation. They do not identify the source used to produce the publication.

## The observation correction is consequential

The historical `_get_obs` builds array views into one shared grid. Writing -1 into blind A's sensory channels also changes any P view overlapping those cells. In particular, if the target lies within A's radius-2 window, its sensory entry is erased for every P whose view overlaps that target. The independent-copy path prevents this cross-agent mutation; it was introduced before the old SoftRole runs, not for these Adam trials. Paper-v1 uses the same corrected view semantics.

A static exhaustive enumeration of all 25×24×23×22 = 303,600 labeled initial layouts gives:

| Initial event | Corrected copies | Historical shared views |
|---|---:|---:|
| At least one P has a positive target channel | 239,008 / 303,600 (78.7246%) | 101,280 / 303,600 (33.3597%) |

The 45.3650-percentage-point difference is independently checked by a closed-form count by target location in `audit.py`. It quantifies the information effect at reset under the actual uniform-distinct distribution. It is **not** a trained-performance estimate, does not rerun policies, and cannot establish how much of the publication gap was caused by this bug because the publication checkout is unknown.

SoftRole additionally merges P/A occupancy counts and appends only each agent's own current capabilities. Count merging removes neighbor-class information from a correct typed observation; independent copying restores intended sensing relative to the buggy shared-view path. These are different changes and should not be compressed into the claim that the entire observation transformation only removes information. Blind A keeps position encoding in the public source and in SoftRole but receives no target/occupancy sensing. Current typed HetNet also has class-specific actor weights and typed communication relations, unlike SoftRole.

## A five-step result is physically plausible

For initial Manhattan distances `d_P0, d_P1, d_A`, every successful policy must satisfy

`T >= max(d_P0, d_P1, d_A + 1)`.

A fully informed controller can attain this by simultaneous shortest paths and A's extra capture action. There are no collisions or prior-P-discovery prerequisites. Over the actual distinct-placement population, exhaustive enumeration gives mean 5.0850593 joint steps and mean team return -0.4 (mean-agent -0.1333333). These are the omniscient population benchmarks, not claimed optima of the partially observed policy class or exact bounds on an arbitrary small finite panel.

For completion steps `tau_i`, successful-episode reward is `-.05 * (sum(tau_i)-3)`, whereas episode length is `max(tau_i)`. Reward and episode length therefore need not rank policies identically. For any episode, `R_team <= -.05*(T-1)`. The paper's PCP pair (-0.364, 9.98) violates the team-sum interpretation of that bound (-0.449), consistent with its different reporting reduction. A result near five steps is consequently not a sign by itself that capture has been removed or the target made dynamic/static differently.

## Which 'HetNet reproduction' is being compared

There are materially different local baselines. The older `stokes_runs/reproduction*/pcp_real` runs use the public-code model/learner lineage; their identities must be retained. The new `logs_1/hetnet-paper-908316_*` stdout declares `paper-v1`, `paper-equations-v1`, `torch-v1`, Adam 0.001, corrected PCP observations, radius 2, 2P1A and the 40M target. It shows a three-layer, class-specific PaperNet. These are the relevant latest training curves, rather than the old reproduction logs.

The new stdout points to `runs/paper_study_20261005_v1/hetnet/...` on Stokes. That research run directory/source/checkpoint tree is not present in the copied `stokes_runs` inventory at this review. Thus its stdout configuration and numerical ledgers can be analyzed, but cannot be freshly bound to those missing checkpoint bytes here. Existing seed-991 paper preflight archives establish the reviewed implementation, not provenance of the missing research checkpoints.

Paper-v1 is a documented reconstruction, not a recovered exact publication checkout. Its Binary convention is 64 bits **per independent head**, hence 256 bits per sender/round, rather than a verified reproduction of the paper's reported 64-total-bit result. The new SoftRole trials are **shared**, one-effective-transform controls with Adam 0.0001; they are not fresh banked/gated trials. Their performance cannot establish a banked-role adaptation benefit. Architecture, information inputs, learner objectives and optimizer recipes still differ between HetNet and SoftRole, so a ranking does not isolate one architectural component.

## Implication for the study

Keep the existing native PCP benchmark and report the result honestly, including cases where current HetNet is better. Do not change physics, reward or episode accounting to make either method win or match a historical number. If nominal 5×5 PCP is nearly saturated, a separately fixed harder condition or earlier capability loss can answer a different question; preserve native results and separate that extension. The current no-failure shared Adam trials by themselves test nominal learning and optimizer sensitivity, not adaptation.

The immediate comparison is a common-budget, matched frozen panel for the newest archived methods, with both team and mean-agent rewards, success, horizon-capped length, all three seeds, declared observation versions, and exact checkpoint hashes. No new training or policy evaluation was performed for this review.
