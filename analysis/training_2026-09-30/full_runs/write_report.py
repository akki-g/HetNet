"""Assemble the amended report from audited summaries and reviewed interpretation."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
S=json.loads((OUT/'summary.json').read_text())

common=['| Task / budget | Model | Success and observed seed range | Team return | Episode length |','|------------|------------------|------------------------:|----------:|----------:|']
for g in S['groups']:
 c=g['common'];a=c['success_rate']
 common.append(f"| {g['task'].upper()} / {g['budget']/1e6:g}M | {g['model']} | {100*a['mean']:.2f}% [{100*a['min']:.2f}, {100*a['max']:.2f}] | {c['team_return']['mean']:.3f} | {c['steps_taken']['mean']:.2f} |")
latest=['| Task | Model | Completed epochs by seed 0 / 1 / 2 | Latest success by seed 0 / 1 / 2 |','|---|---|---|---|']
for task in ['pp','pcp','fc']:
 for model in ['HetNet Real','SoftRole shared','SoftRole banked']:
  rs=sorted([r for r in S['runs'] if r['primary'] and r['task']==task and r['model']==model],key=lambda r:r['seed'])
  successes=' / '.join(format(100*r['latest50']['success_rate'], '.2f')+'%' for r in rs)
  latest.append(f"| {task.upper()} | {model} | {' / '.join(str(r['epochs']) for r in rs)} | {successes} |")
# Per-run table preserves progress and exact boundaries in the companion CSV.
runrows=['| Task and model | Seed | Epochs | Steps in millions | Episodes | Latest mean length |','|------------------|----:|------:|------------:|----------:|------------:|']
for r in S['runs']:
 if r['primary']:
  runrows.append(f"| {r['task'].upper()} {r['model']} | {r['seed']} | {r['epochs']} | {r['steps']/1e6:.3f} | {r['episodes']:,} | {r['latest50']['steps_taken']:.2f} |")

text=r'''# HetNet and SoftRole Training Review

Full run artifact amendment, 30 September 2026

**The full files strengthen confidence in the recorded training progress, but do not yet establish capability adaptation or a benefit from expert banks.** Shared SoftRole remains ahead of banked on PCP at comparable sample budgets. Banked is improving: the shared-minus-banked success gap falls from 14.7 percentage points near four million steps to 9.2 points near five million. FC remains the weakest domain and deserves targeted diagnosis. Its banked gates show little variation across the observed trajectories, and all recorded SoftRole updates are clipped; neither observation alone establishes an optimization failure.

This amendment replaces the earlier stdout-only interpretation as the current report. It examines the full files in `stokes_runs/runs/`: configurations, source records, checkpoints, update logs and episodes. The [historical stdout report](stdout_snapshot.md) remains available with an explicit correction to its termination discussion. The [PDF report](training_report.pdf) renders the equations and figures; [machine-readable summaries](full_runs/summary.json) retain exact values and window boundaries.

The main unresolved scientific question is now sharper: **does conditional mixing improve performance when capabilities or team composition change, beyond what a capability-aware shared recurrent policy already achieves?** Current runs measure fixed-team nominal training. They cannot answer that question, and stronger nominal scores should not be presented as adaptation results.

## 1. Evidence coverage and what changed

There are 37 run folders, but only **27 primary runs**: nine `reproduction-fast` runs and eighteen `softrole_primary` runs, covering three tasks and three training seeds per method. The nine shorter `reproduction` launches and one `speed-check` reproduce corresponding primary prefixes exactly in model signatures and numerical metrics, apart from wall time. They corroborate repeatability of those prefixes; they are not additional independent seeds.

| Primary ledger | Completed epochs | Environment steps | Episodes |
|---|---:|---:|---:|
| HetNet reproduction | 5,060 | 113,757,971 | 2,361,458 |
| SoftRole completed epochs | 6,315 | 139,063,877 | 8,459,504 |
| Combined completed epochs | 11,375 | 252,821,848 | 10,820,962 |

SoftRole's episode files contain **8,475,482 episodes and 139,223,160 steps**, including records beyond the last completed epoch. These larger totals must not be substituted into the completed-epoch table. All 63,214 available update records and all completed SoftRole epoch records reconcile with episode counts and steps; completed epochs also match outcome, gate and null-attention means. Individual policy/value losses are not present in the episode ledger and cannot be reconstructed from it. No scheduled or exposed failures or communication/gate interventions occur in the supplied training data. [Episode audit](full_runs/softrole_audit/findings.json)

The pull is not an atomic snapshot. For example, FC banked seed 0 has metrics and episodes through update 3190, while its update file stops at 3189. Other files contain differing trailing prefixes. Completed metrics remain usable because they reconcile with episodes; gradient summaries use the records actually present and never invent the missing update. A copied tail cannot certify current scheduler status or successful completion of the planned budget.

All 4,860 completed reproduction epoch blocks from the earlier stdout pull match the corresponding new stdout blocks byte-for-byte. SoftRole configuration/source identities are now observed rather than inferred from filenames. The recorded configuration is fixed 3P0A for PP and 2P1A for PCP/FC, with `failure_prob=0`, no composition mixture and no held-out panel. The analysis therefore extends the earlier runs rather than treating additional files as new experimental replicates. [Reproduction audit](full_runs/reproduction_audit/findings.json), [provenance audit](full_runs/provenance_audit/audit.json)

**Checkpoint integrity is strong within the checked scope.** All 120 SoftRole checkpoints pass actual tensor hashes, sidecar/embedded signatures, configuration/source identity, finite model/optimizer state and recorded training-count checks. All 97 primary reproduction checkpoints pass actual tensor hashes, finite-state and optimizer-step checks; the additional speed-check checkpoint has matching sidecar/file records. Changing signatures establish changing parameters, not improvement of the policy. [SoftRole verification](full_runs/provenance_audit/audit.json), [reproduction tensor verification](full_runs/reproduction_audit/checkpoint_verification.json)

Reproduction records source commit `37dd2f3`, an empty tracked-source patch, Python 3.12.7, Torch 2.2.1, DGL 2.1.0 and NumPy 1.26.4. All eighteen SoftRole archives share source digest `35e7fc3a53d6…` at commit `6994d87`; all 73 files in every archive match their manifest. Its run records specify Torch 2.2.1+cu121, NumPy 1.26.4 and `corrected-observation-v1`, but not a full host/Python/package inventory. Legacy model tensors mix float64 and float32 attention parameters; SoftRole tensors are float64 throughout. These distinctions preclude describing the two pipelines as numerically identical or making hardware-normalized speed claims.

## 2. Comparison method and mathematical scope

The structured reproduction recorder supplies correct sample counts. Its stdout counter bug still exists: for ten batch counts \(n_i\), the displayed increment is

\[
C=\sum_{b=1}^{10}\sum_{i=1}^{b}n_i
 =\sum_{i=1}^{10}(11-i)n_i,
\qquad S=\sum_{i=1}^{10}n_i.
\]

The recorder instead adds each fresh batch once. We use its \(S\), not the printed \(C\). Division by 5.5 is not a general exact correction to stdout; it requires \(\sum_i(5.5-i)n_i=0\). Equal batch sizes suffice, but are not necessary. [Recorder source](../../hetnet_ext/recording.py)

For each task, let \(B\) be the largest multiple of 500,000 steps not exceeding the shortest primary run. This gives **5M steps for PP/PCP and 5.5M for FC**. For each run, select its last 50 completed epochs ending at or before \(B\). If \(N_{se}\) is the episode count and \(y_{se}\) the epoch mean, report

\[
\widehat y_s(B)=
\frac{\sum_{e\in W_s(B)}N_{se}y_{se}}
     {\sum_{e\in W_s(B)}N_{se}},
\qquad
\widehat y(B)=\frac{1}{3}\sum_{s=0}^{2}\widehat y_s(B).
\]

The first identity exactly reconstructs the pooled episode mean within that run's selected window. Equal weighting across runs prevents a faster-finishing policy from receiving more weight simply because it generated more episodes. These are **training-window statistics for changing policies**, not estimates from a frozen final policy.

This is approximate sample-budget matching, not exact boundary alignment. Selected endpoints lie at 4.983–4.999M for PP, 4.979–4.998M for PCP, and 5.477–5.499M for FC. The largest shortfall is below 0.42% of the task budget. Fifty epochs also span somewhat different sample intervals: starts are approximately 3.948–3.995M, 3.926–3.946M and 4.216–4.259M, respectively. Exact boundaries and episode counts are in [run_summary.csv](full_runs/run_summary.csv). No outcomes are interpolated to pretend a boundary was observed.

The earlier report used equal-epoch means because reproduction episode counts were unavailable. This report deliberately switches both pipelines to episode weighting. Thus updated values reflect **later data, a different window and improved weighting**; the old 15.3-point PCP gap should not be compared directly with the new 9.2-point gap as a longitudinal estimate. The 14.7-to-9.2 comparison uses the new estimator consistently at 4M and 5M.

Seed ranges below are observed extrema, **not confidence intervals**. Three training seeds provide limited information about run-to-run uncertainty; millions of correlated, changing-policy training episodes do not create millions of independently trained models. Original collectors also use overlapping numerical worker seeds across adjacent training seeds, so an IID-seed interpretation is particularly questionable there. We make no significance or equivalence claim from these ranges. This follows the evaluation concerns documented by [Agarwal et al.](https://arxiv.org/abs/2108.13264) and [Henderson et al.](https://arxiv.org/abs/1709.06560).

## 3. Performance at common sample budgets

{{COMMON_TABLE}}

Within-task SoftRole configurations differ only in model variant and training seed. The comparison is therefore a useful practical test of these implemented variants. It does not isolate adaptive routing from capacity: banked adds 53,084 parameters, giving 224,171 versus 171,087 for PP/PCP, and 166,827 versus 113,743 for FC. Original HetNet is a contextual reference: its Real communication, typed/legacy observation path, recurrence, critic and optimization differ from SoftRole's binary, corrected, untyped pathway. A cross-pipeline advantage cannot be attributed to the gate alone. The original domain/method reference is [Seraj et al., AAMAS 2022](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf).

![Figure 1. Outcomes against exact recorded environment steps. Individual runs are retained; smoothing uses episode-weighted trailing windows.](full_runs/sample_learning_curves.png)

![Figure 2. Common-budget comparisons. Dots are individual run means; black diamonds are equal-run means. Lines are observed ranges, not confidence intervals.](full_runs/common_sample_comparison.png)

**PP has little remaining discrimination in success, but still has discrimination in cost and completion time.** Both SoftRole variants have 100% observed success in the 5M window, while shared averages 4.94 steps and banked 5.02. The latest windows are also nearly saturated: one shared seed records one unsuccessful episode among 207,356, and the other five windows have none. Latest length medians are five steps and 99th percentiles eight. These observations do not prove zero future failure probability, optimality or generalization. They do support shifting diagnostic attention from rounded success percentages to length, return and failure cases.

**PCP currently favors shared in sample efficiency; banked is still learning.** At 5M, shared has 91.07% success versus banked's 81.89%, with better return and shorter episodes. Every shared run mean exceeds every banked run mean at this budget. At 4M the corresponding means are 84.67% and 70.01%. Banked's later latest-window success improves relative to its preceding 50 epochs in all three seeds; its latest values are 83.01%, 82.92% and 94.58%. This supports a learning-speed disadvantage over the observed budgets, not a claim of an asymptotically inferior solution or a diagnosed optimization cause.

The algebra explains why representational capacity alone cannot settle this result. A bank satisfies

\[
B(x,g)=\sum_k g_k(W_kx+b_k).
\]

If every expert equals \((W,b)\), then \(B(x,g)=Wx+b\) because \(\sum_k g_k=1\). Banked can reproduce a shared transform; finite optimization need not find equally good parameters. Constant-gate and capability-only controls are already available to distinguish conditional routing from overparameterized optimization. Shared itself remains capability-aware and recurrent, consistent with the adaptive-teaming precedent in [Howell et al.](https://proceedings.mlr.press/v229/howell23a.html).

**FC remains unresolved, without a demonstrated bank advantage at the common budget.** Shared and banked are at 47.15% and 47.31% success, a descriptive difference of only 0.16 percentage points. This is not evidence of statistical equivalence. HetNet's 31.72% mean at that budget is strongly influenced by seed 0's prolonged dip: its three values are 17.18%, 36.81% and 41.17%. At later, unequal endpoints, all three HetNet runs reach approximately 40% success. Seed 0's latest return now improves over its first window, so the earlier snapshot's statement that all three FC returns worsened no longer applies.

SoftRole FC also has a recent regression worth following: shared seed 0 falls from 43.48% success in its preceding 50 epochs to 39.33% in its latest 50, with mean return moving from −217.60 to −228.50. Other seeds behave differently. Across the latest FC windows, approximately 49.6–60.7% of shared episodes and 51.4–53.5% of banked episodes reach the 300-step horizon. This is a persistent task-solving limitation, not merely small numerical movement in an optimizer loss. The logs do not reveal whether navigation, discovery, communication, action selection or critic fitting is the dominant cause.

For context, the full endpoint table follows. **Latest-window values below are progress diagnostics, not matched-budget rankings.** PP/PCP target 2,000 epochs and FC targets 1,400; all runs remain partial relative to those configured budgets.

{{LATEST_TABLE}}

## 4. Optimization diagnostics and what the losses mean

**All 63,214 recorded SoftRole updates are clipped.** The smallest observed preclip norm is 10.51, against a threshold of 0.75. Norms are measured after global episode averaging. The applied transformation is

\[
c_t=\min\!\left(1,\frac{0.75}{\|g_t\|_2+10^{-6}}\right),
\qquad \widetilde g_t=c_tg_t.
\]

Latest-window median norms are approximately 24–25 for PP shared and 30–43 for PP banked; 101–353 and 520–561 for PCP; and 30,819–36,906 and 21,299–28,326 for FC. FC median multipliers are therefore on the order of \(2\times10^{-5}\) to \(4\times10^{-5}\). This establishes that clipping is a routine part of the implemented update, not an occasional safeguard. It does **not** establish exploding gradients or that clipping caused the performance gap. Gradient clipping has a well-established motivation for recurrent optimization, but its benefit here requires a controlled test. [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.html)

RMSprop then applies coordinatewise preconditioning, schematically

\[
v_t=0.97v_{t-1}+0.03\widetilde g_t^2,
\qquad
\theta_{t+1}=\theta_t-
10^{-4}\frac{\widetilde g_t}{\sqrt{v_t}+10^{-6}}.
\]

Consequently, clipping does not bound the parameter-step norm by \(0.75\times10^{-4}\), nor does a tiny multiplier translate directly into that fraction of an effective learning rate. The actual parameter displacement, actor/critic gradient contributions and their alignment are not recorded per update. These are sensible next diagnostics if FC remains weak. [PyTorch 2.2.1 RMSprop source](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/optim/rmsprop.py), [clipping source](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/nn/utils/clip_grad.py)

**The two pipelines' raw value-loss numbers are not comparable.** The original collector returns an episode-averaged sum of classwise time-averaged MSEs, \(L_c\). Its recorder logs

\[
V_{\mathrm{log}}=\frac{\sum_{c=1}^{40}L_c}{S_{\mathrm{epoch}}}.
\]

Thus \(V_{\mathrm{log}}S_{\mathrm{epoch}}/40\) recovers the unweighted mean of 40 collector losses; the displayed quantity itself is neither conventional episode-mean nor transition-mean MSE. Even undoing this denominator does not align its classwise targets with SoftRole's scalar team target. Also, the legacy resolved argument `value_coeff=.01` is unused on the exercised HetGAT path: the executed source uses actor coefficient 50 and critic coefficient 1. [Exercised loss implementation](../../hetgat/policy.py) and [recorder](../../hetnet_ext/recording.py), checked at recorded commit `37dd2f3`; [numerical audit](full_runs/reproduction_audit/findings.json)

SoftRole logs the episode mean of

\[
L_V=\frac1T\sum_{t=0}^{T-1}
\left(V_t-\operatorname{stopgrad}(\widehat A_t+V_t)\right)^2.
\]

FC's first-50 means are approximately 327–484 and latest-50 means 1,055–1,335. The policy changes both the visited states and the value targets; multiplying targets and predictions by a constant \(a\) multiplies squared error by \(a^2\). This proves why absolute loss scale alone is insufficient to compare domains. The rise warrants diagnosis but does not prove divergence. GAE also trades bias and variance with an approximate critic. [Schulman et al.](https://arxiv.org/abs/1506.02438)

The actor surrogate sums over episode time, whereas value MSE averages over time; the actor also carries coefficient 50. FC's longer episodes and different reward scale therefore matter to gradient magnitude. The signed actor loss approaching zero is not a convergence test. Finite checkpoints and changing weights rule out some failures, but do not establish good gradients, value calibration or reliable exploration.

Some original model tensors remain identical to initialization. This is not automatically a stalled optimizer: absent relations and singleton attention neighborhoods can make parameters structurally inactive. With one candidate, \(\alpha=e^z/e^z=1\), hence \(\partial\alpha/\partial z=0\). The audit identifies relevant absent/singleton relations; it does not assume this explains every unchanged tensor.

![Figure 3. SoftRole preclip norms, applied multipliers and value-target MSE. Logarithmic scales are explicit.](full_runs/optimization_diagnostics.png)

## 5. What the communication diagnostics reveal

Mean gate entropy alone cannot distinguish a diffuse fixed mixture from changing routing. Episode records provide both average entropy and average expert weights, allowing the exact decomposition

\[
J=H(\bar g)-\mathbb E[H(g)]
 =\mathbb E[D_{\mathrm{KL}}(g\Vert\bar g)]\ge0,
\qquad \bar g=\mathbb E[g].
\]

Expanding KL proves the identity: \(\mathbb E\sum_k g_k\log g_k-\sum_k\bar g_k\log\bar g_k\). Here the expectation chooses an episode uniformly, then an agent-time uniformly within that episode, matching the stored averaging. A 50-epoch window mixes changing policies, so \(J\) includes parameter drift as well as differences between histories, agents and timesteps.

Latest-window FC banked \(J\) values are **0.003044, 0.002175 and 0.001309 nats**; mean gate entropies are 1.1455, 1.3080 and 1.3669 nats, compared with \(\log4\approx1.3863\). Thus broad mixtures coexist with little observed routing variability. PCP values of \(J\) are 0.0943–0.1341, and PP values are 0.2213–0.3314. These are diagnostic differences, not evidence of semantic roles or successful capability adaptation. [Audited gate summaries](full_runs/softrole_audit/findings.json)

If a gate were exactly constant, each affine bank would reduce to one fixed effective matrix and bias. Small gate variation alone does not establish functional equivalence, because

\[
\|B(x,g)-B(x,\bar g)\|
\le\sum_k|g_k-\bar g_k|\,\|W_kx+b_k\|.
\]

Expert-output magnitudes can amplify small weight changes. A constant-gate control and within-model gate interventions are required before concluding that FC's banks are unnecessary. Expert indices also have no common semantic alignment across seeds; the plot deliberately keeps each run's expert weights separate.

Null attention is similarly descriptive. Because real-sender weights are nonnegative and sum to \(1-\alpha_{\mathrm{null}}\), the triangle inequality gives

\[
\left\|\sum_k\alpha_km_k\right\|
\le(1-\alpha_{\mathrm{null}})\max_k\|m_k\|.
\]

For an empty neighborhood, the aggregate is exactly zero. Null mass bounds received magnitude only when message norms are considered; it does not measure causal communication usefulness, information transmitted or recovery ability. Those require interventions and outcomes.

![Figure 4. Gate entropy, the entropy decomposition and null attention over training.](full_runs/gate_diagnostics.png)

![Figure 5. Pooled expert weights remain separate by seed. Nonuniform weights alone are not role specialization.](full_runs/expert_weights.png)

## 6. Consequences for the capability adaptation study

**The present data do not yet test the project's central adaptation claim.** On every observed fixed-team training episode, sensing agents have \(\kappa=(1,0)\) and actuators have \((0,1)\). These capability vectors identify physical class perfectly on this support. Removing explicit class labels and class-indexed weights is an architectural fact; nominal metrics cannot demonstrate behavior beyond this class-correlated support. Known sensor loss introduces \((0,0)\), and composition changes test a different team distribution.

The full episode records provide a useful preflight diagnostic. For a nominal episode of length \(T\) and independent zero-indexed event time \(U\sim\mathrm{Uniform}\{10,\ldots,30\}\), the event is reached exactly when \(T>U\). Counting eligible event times gives

\[
q(T)=\Pr(U<T\mid T)
=\frac{\max(0,\min(21,T-10))}{21}.
\]

Applied to each current PCP run's latest 50 epochs:

| Nominal event exposure diagnostic | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Shared | 57.42% | 12.64% | 21.14% |
| Banked | 75.98% | 75.44% | 61.89% |

With 50% event assignment, expected exposure on these unchanged nominal prefixes is half these values. This is **not** a frozen-checkpoint evaluation, nor a prediction of post-failure success; the windows contain different training stages and policies. It does establish that the existing time window can yield few exposed episodes for fast policies. [Exposure derivation and values](full_runs/softrole_audit/nominal_failure_exposure.json)

A second risk remains after exposure: the target is stationary, memory survives failure, and reached agents cannot move away. A victim may already know the target location or have completed its navigation duty. The current episode summaries do not record target visibility/history or reached status at a hypothetical event; duration cannot establish how consequential the failure would be.

**Correction to the earlier report:** under the released static-target PP/PCP rules, \(\mathrm{reached}_i=1\) implies the agent remains at the target. Natural termination requires all reached and required captures completed, hence success. Unsuccessful default episodes run to the horizon. All 8.48 million audited SoftRole episode records are consistent with unsuccessful episodes reaching their respective horizon. Average episode length still combines successful completion and failed horizon-length episodes; success-only averages would omit failures. [Movement and termination source](../../envs/ic3net_envs/predator_capture_env.py)

The immediate preparation should be a training-composition pilot using frozen checkpoints, matched failure/sham scenarios, exposure reporting and examination of the victim's pre-event situation. Preserve the declared event schedule as the reference; any altered timing or harder supplementary environment must be declared before held-out evaluation. Do not select failures or victims to favor banked. PCP sensor loss is already implemented, while FC loss requires discovery/reward semantics to respect working sensors. Actuation failures and mid-episode arrivals/departures remain separate environment work.

Report both absolute failure performance and the failure-minus-sham contrast; a weak nominal policy can have a small degradation simply because its baseline is low. Preserve the full assigned scenario panel for the primary comparison. Conditioning on exposure can select different subsets when policies have different completion times. Gate freezing with matched shams can then test whether gate changes contribute specifically under failure; it does not freeze the entire recurrent policy or prove role discovery.

![Figure 6. Episode survival curves explain event exposure. The shaded PCP interval is the planned event-time range, not an observed failure experiment.](full_runs/episode_length_survival.png)

## 7. Questions these artifacts answer and questions they leave open

| Research question | Evidence-supported answer | Remaining evidence |
|---|---|---|
| Are the runs numerically progressing with coherent records? | Yes within the audited records: exact ledger reconciliation, finite checkpoint states and changing model signatures. | This is not validation of every future update or proof of useful gradients. |
| Does shared beat banked on nominal PCP at current budgets? | Yes descriptively at both 4M and 5M steps; the gap narrows and banked continues learning. | Completed budgets, frozen evaluations and more independent training seeds. |
| Does banked improve FC? | No clear common-budget advantage is established; latest endpoints are unequal and seed behavior differs. | Controlled evaluations and mechanism ablations. |
| Is FC failing because its gradients explode or its critic is worse than HetNet's? | The logs do not establish either cause. Clipping is ubiquitous and raw critic losses have incompatible definitions. | Target/prediction statistics, separate gradient components, actual parameter-update norms and controlled training-distribution tests. |
| Have semantic roles or adaptive routing been learned? | Gate distributions vary, but FC variation is especially small; neither proves semantic roles. | Behavioral associations, interventions, constant/capability controls, and fixed-policy measurements. |
| Have we demonstrated sensor recovery or composition transfer? | No. There are zero events and only fixed original teams. | Frozen matched event panels and held-out team evaluations. |
| Have we reproduced the paper's reported performance? | Domain recipes, code provenance and numerical integrity are documented; that is narrower than reproducing a published performance estimate. | Matched published evaluation definitions/budgets and uncertainty, including the changed observation contract in SoftRole. |

The defensible current contribution is an implemented, auditable reformulation and a controlled finding that extra conditional banks have not improved nominal sample efficiency on PCP. A robust simpler model would remain a valid outcome. A claimed advantage in adaptation must be earned by the next experiments, not inferred from nominal training or the existence of a gating mechanism.

## 8. Recommended decisions

1. **Keep shared as a primary research candidate.** Complete the declared comparison or document any common-budget stopping rule before further model selection. Retain all seeds and unsuccessful cases. Do not tune on held-out compositions to rescue banked.
2. **Treat FC as a diagnostic priority.** Inspect frozen-policy behavior and existing checkpoints; separate failures in exploration, coordination and actuation. If additional instrumentation is needed, record value targets/predictions, actor/critic gradient components and parameter displacements. None of these causes is identified by current scalar logs.
3. **Run the capability pilot before a large failure sweep.** Audit actual event exposure and whether the victim still needs fresh sensing. Then compare shared and banked under the same declared failure-training distribution, with no-event and matched-sham evaluations. Keep unexpected-failure transfer from nominal training separate from failure-trained adaptation.
4. **Use the existing mechanism controls.** Constant gates test fixed-mixture optimization; capability-only gates test the need for history-dependent gating; event-timed freezes and communication removal test within-policy dependence. None alone proves a unique explanation.
5. **Improve the next artifact handoff.** Copy an immutable checkpoint plus configuration/source identity and an explicit common record cutoff. Archive evaluator source and a complete runtime inventory. The present trailing-file discrepancy is manageable, but should not become an undocumented convention.

The research protocol already proposes an initial three-seed screen, then more independent seeds where compute permits and common evaluation scenarios per composition. More seeds improve the basis for uncertainty assessment; no fixed seed count guarantees power. The present report intentionally avoids inferential confidence intervals for changing-policy training curves. Literature supports the evaluation principles; the empirical claims here come from the supplied artifacts, not from assuming another paper's results transfer.

## 9. Reproducibility and complete primary inventory

The [analysis script](full_runs/analyze_full.py) exports [epoch metrics](full_runs/epoch_metrics.csv), [run summaries](full_runs/run_summary.csv) and [structured input hashes](full_runs/structured_input_hashes.json). Separate reproducible audits cover [reproduction](full_runs/reproduction_audit/audit.py), [SoftRole episodes](full_runs/softrole_audit/audit.py), and [source/checkpoints](full_runs/provenance_audit/audit.py). Episode hashes were collected during streaming, avoiding a second full scan. The [episode diagnostic script](full_runs/plot_episode_diagnostics.py) uses the audited sufficient statistics. The [PDF build script](full_runs/build_pdf.py) typesets the report with Pandoc and Tectonic.

No training, frozen policy evaluation, model modification or checkpoint modification was performed for this report. Plotting and document conversion use isolated tool environments. All comparison numbers were independently checked; input files were treated read-only. Figures are also available individually and in [training plot](full_runs/full_run_plots.pdf) and [episode diagnostic](full_runs/episode_diagnostic_plots.pdf) PDF collections.

{{RUN_TABLE}}

The [historical stdout snapshot](stdout_snapshot.md) retains the earlier numerical analysis. Its narrower information limits and superseded interpretation should not be read as findings from this expanded artifact set.
'''
text=text.replace('{{COMMON_TABLE}}','\n'.join(common)).replace('{{LATEST_TABLE}}','\n'.join(latest)).replace('{{RUN_TABLE}}','\n'.join(runrows))
(OUT.parent/'report.md').write_text(text)
print('Wrote amended report:',len(text.split()),'words')
