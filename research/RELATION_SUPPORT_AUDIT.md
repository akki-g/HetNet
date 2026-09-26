# Relation support in source training and frozen transfer

**Prospective interpretation, 26 September 2026; recorded before frozen outcomes.** The 2P1A source recipe
leaves the entire A→A message branch unused and gives several attention
relations only one incoming sender. Frozen composition transfer therefore
changes which parameter groups affect the output, in addition to changing
team size. This is an interpretation constraint for the existing F-comp
conditions, not a change to the recipe, conditions, or architecture.

## Source-derived support

The executed collection path calls `build_hetgraph` with
`with_self_loop=False` and two state nodes
([policy.py:516–525](../hetgat/policy.py#L516)). With the default unlimited
communication ranges, the graph contains all directed cross-agent edges
except self edges ([utils.py:64–92](../hetgat/utils.py#L64)); P agents send to
state node 0 and A agents to state node 1
([utils.py:138–146](../hetgat/utils.py#L138)). The two real-valued layers use
the same graph ([uavnet.py:102–109](../hetgat/uavnet.py#L102),
[367–368](../hetgat/uavnet.py#L367)).

For a relation r, head h, and receiving node i, the implementation computes

\[
z_{ji}^{r,h}=\operatorname{LeakyReLU}\!\left[
(a_{\mathrm{src}}^{r,h})^\top m_j^{r,h}
+(a_{\mathrm{dst}}^{r,h})^\top s_i^h\right],\qquad
\alpha_{ji}^{r,h}=
\frac{\exp z_{ji}^{r,h}}{\sum_{k\in\mathcal N_r(i)}\exp z_{ki}^{r,h}},
\quad o_i^{r,h}=\sum_{j\in\mathcal N_r(i)}\alpha_{ji}^{r,h}m_j^{r,h}.
\]

Here m is the relation-specific affine message transform and s is the
receiving node's self transform. Normalization is per receiving node **within
one relation**, as shown by the separate `edge_softmax(g['p2p'], ...)`,
`edge_softmax(g['a2p'], ...)`, and `edge_softmax(g['a2s'], ...)` calls
([fastreal.py:197–211](../hetgat/graph/fastreal.py#L197),
[240–254](../hetgat/graph/fastreal.py#L240),
[288–302](../hetgat/graph/fastreal.py#L288)). The DGL 2.1.0
[implementation at lines 12–28](https://github.com/dmlc/dgl/blob/v2.1.0/python/dgl/ops/edge_softmax.py#L12-L28)
declares `norm_by="dst"` and specifies
incoming-edge normalization. The self transform is added outside this
relation softmax ([fastreal.py:311–345](../hetgat/graph/fastreal.py#L311)); it
does not create a second competitor in a singleton relation.

For a singleton neighborhood, alpha=1 and the softmax Jacobian is

\[
\frac{\partial\alpha_j}{\partial z_k}
=\alpha_j(\mathbf 1\{j=k\}-\alpha_k)=0.
\]

Consequently the relation's dedicated source/destination attention vectors
receive zero loss gradient. Its message transform can still learn through
o=m. For an empty relation, the attention/message-reduction branch is skipped
and contributes nothing. In particular, A→A is skipped at
[fastreal.py:256–270](../hetgat/graph/fastreal.py#L256) and
[330–331](../hetgat/graph/fastreal.py#L330); its affine transform, although
evaluated earlier, has no path to the returned output.

The following are incoming degrees **per receiver**, with state columns
referring to the corresponding class's state node:

| Composition | P→P | P→A | A→P | A→A | P→state | A→state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2P1A source | 1 | 2 | 1 | 0 | 2 | 1 |
| 3P1A | 2 | 3 | 1 | 0 | 3 | 1 |
| 2P2A | 1 | 2 | 2 | 1 | 2 | 2 |
| 3P3A | 2 | 3 | 3 | 2 | 3 | 3 |
| 4P6A | 3 | 4 | 6 | 5 | 4 | 6 |

Thus a fresh fixed-2P1A run supplies no gradient to eight A→A tensors: for
each `layer{1,2}.gat_conv`, these are `a2a_src`, `a2a_dst`,
`fc.a2a.weight`, and `fc.a2a.bias`. It also gives exactly zero mathematical
gradient to the twelve `p2p_src/dst`, `a2p_src/dst`, and `a2s_src/dst`
tensors across both layers. The source recipe's RMSprop has no weight decay
or momentum configured ([trainer.py:55–56](../trainer.py#L55)); no update
path changes these unused/zero-gradient parameters in this fresh-training
setting. This does not say their input representations stay fixed: other
message transforms and the recurrent encoders learn.

## Existing evidence, without new experiments

This audit reread the preserved
[2P1A](../evidence/gate_a/hetnet_gradient_repaired_2P1A_storage_20260926_01.json)
and
[4P6A](../evidence/gate_a/hetnet_gradient_repaired_4P6A_storage_20260926_01.json)
gradient diagnostics and their three NPZ siblings per composition. All six
NPZ hashes matched their JSON records. These are seed-17 structural fixtures
with batch 4, horizon 4, three updates, and four processes, not final trained
policies or performance results. They use Torch 2.2.1, the approved gradient
storage correction, unlimited ranges, vision 2, and no communication loss.

The source model has 76 named parameter tensors; only 68 have a non-null
gradient, with exactly the eight A→A tensors listed above absent. Every
one of the twelve singleton-relation attention tensors has gradient norm
**exactly zero in all 12 process/update snapshots**. All twenty unsupported
tensors have identical hashes across the three post-step signatures.
The corresponding message weights receive nonzero gradients:

| Source message weight | Layer 1 gradient norm range | Layer 2 gradient norm range |
| --- | ---: | ---: |
| `fc.p2p.weight` | 0.00832893–0.02091998 | 0.02757822–0.07199376 |
| `fc.a2p.weight` | 0.00648671–0.01316048 | 0.02093010–0.05477643 |
| `fc.a2s.weight` | 0.000149664–0.000224006 | 0.00106244–0.00118121 |

These are Euclidean norms of each flattened **fresh, locally clipped**
gradient, before inter-process summation and step normalization. The bias
gradients are also nonzero in all 12 snapshots. The independently trained
4P6A fixture has non-null gradients for all 76 tensors, including the A→A
weights (layer-1 norms 0.00959668–0.02304275; layer-2 norms
0.05791830–0.12319823). This confirms support in that fixture; it is not a
claim that any relation learned a useful communication behavior.

The first completed full-gate single-process source smoke provides a longer
engineering check: the same eight A→A tensors and twelve singleton score
tensors remain byte-identical from initialization through 30 optimizer
updates (three epochs), while 54 other tensors change. Its
[signature comparison](../evidence/gate_a/source_relation_support_smoke_20260926_02.json)
records both input-file hashes and the raw run path. This check uses existing
Gate A artifacts, not an added training experiment or a transfer outcome.

The dtype is mixed despite the float64 default: attention vectors are
explicitly created with `torch.FloatTensor`
([fastreal.py:80–96](../hetgat/graph/fastreal.py#L80)) and are float32 in the
recorded signatures; affine message parameters are float64. Norms above
were computed in float64 from the saved arrays. Nonzero values near numerical
roundoff in some non-singleton destination-score gradients must not be
interpreted as substantive learning. Multiple incoming edges permit score
learning but do not guarantee nonzero gradients for every feature/context.

## Interpretation of the existing frozen comparisons

- **2P1A→3P1A:** A→A stays absent. P→P changes from one to two incoming
  senders, making the source-untrained P→P attention vectors potentially
  consequential. Its message transform was trained. P→A and P→state also
  change from two to three neighbors.
- **2P1A→2P2A:** A→A becomes present and adds a message transform never
  trained on the source task. Its attention coefficient is still 1 because
  each A receives from only one other A, so the A→A attention vectors remain
  irrelevant to that coefficient. A→P changes from one to two senders,
  making its source-untrained attention vectors potentially consequential;
  the A→P message transform was trained. P→P remains singleton.
- **2P1A→3P3A/4P6A:** both effects occur, and A→A now has multiple incoming
  senders, activating its source-untrained attention vectors as well as its
  source-untrained affine message transform.

A→state attention also gains support when A count increases, but state nodes
have no outgoing edges to agents
([utils.py:133–152](../hetgat/utils.py#L133)). Their final outputs feed the
critic heads ([uavnet.py:370–391](../hetgat/uavnet.py#L370)); with frozen
weights and actions drawn from the P/A logits, this critic-only attention
change has no direct path into the action distribution. Its role during
native training is different because critic losses can shape shared upstream
representations.

Accordingly, a frozen/native performance gap would quantify transfer under
these simultaneous changes. It would not by itself identify failed role
inference, attention dilution, or the new A→A branch as its cause. The existing
3P1A and 2P2A conditions differ in which source-unsupported paths become
active, so they should retain separate interpretations. Native training
provides the destination's available gradient support, but cannot guarantee
that all supported parameters learn useful behavior. Permutation equivariance
does not remove any of these support changes. These conclusions assume the
declared graph, finite logits, fixed source roster, and no additional
regularizer, parameter tying, optimizer state, or graph intervention.
