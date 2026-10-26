"""Independent-head HetGAT for the explicitly reconciled paper-v1 model.

The paper's Eq. 2--5 and section 4.3 specify typed attention and independent
parallel heads. Unlike fastbinary's shared payload, each Binary head here owns
its encoder, categorical binarizer and receiver-class decoder. The chosen
study convention is 64 bits PER HEAD, or 256 bits per sender per round; this is
an explicit reconstruction choice, not an identified publication checkpoint.

Both backends use the same parameter containers and sender computations. DGL
and torch-v1 differ only in the relation aggregation implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from hetgat.graph.torch_backend import RELATIONS, aggregate


class PaperHead(nn.Module):
    def __init__(self, in_dim, out_dim, binary, msg_dim):
        super().__init__()
        transforms = {
            "P": ("P", "P"), "A": ("A", "A"),
            "p2s": ("P", "state"), "a2s": ("A", "state"),
            "in": ("state", "state"),
        }
        if not binary:
            transforms.update({name: (src, dst) for src, name, dst in RELATIONS[:4]})
        self.fc = nn.ModuleDict({name: nn.Linear(in_dim[src], out_dim[dst])
                                for name, (src, dst) in transforms.items()})
        self.binary = binary
        if binary:
            self.encoder = nn.ModuleDict({key: nn.Linear(in_dim[key], msg_dim)
                                          for key in ("P", "A")})
            self.binarize = nn.Linear(1, 2)
            self.decoder = nn.ModuleDict({key: nn.Linear(msg_dim, out_dim[key])
                                          for key in ("P", "A")})
        # Preserve the reconstruction's explicitly mixed attention dtype.
        self.attention = nn.ParameterDict()
        for _, name, target in RELATIONS:
            for side in ("src", "dst"):
                parameter = nn.Parameter(torch.empty(1, out_dim[target], dtype=torch.float32))
                nn.init.xavier_normal_(parameter, gain=nn.init.calculate_gain("relu"))
                self.attention[name + "_" + side] = parameter

    def payloads(self, features):
        """Draw P then A once per head, broadcasting each bit vector to neighbors."""
        bits = {}
        for key in ("P", "A"):
            encoded = self.encoder[key](features[key])
            logits = self.binarize(encoded.unsqueeze(-1))
            categorical = F.gumbel_softmax(logits, tau=1, hard=True, dim=-1)
            bits[key] = categorical @ categorical.new_tensor([0, 1])
        return bits


class PaperLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, msg_dim=64,
                 use_real=True, merge="cat", message_backend="dgl"):
        super().__init__()
        if message_backend not in ("dgl", "torch-v1"):
            raise ValueError("Unknown message backend: " + str(message_backend))
        if merge not in ("cat", "avg"):
            raise ValueError("Head merge must be cat or avg")
        self.in_dim, self.out_dim = dict(in_dim), dict(out_dim)
        self.num_heads, self.msg_dim = num_heads, msg_dim
        self.use_real, self.merge = use_real, merge
        self.message_backend = message_backend
        self.heads = nn.ModuleList([
            PaperHead(in_dim, out_dim, not use_real, msg_dim) for _ in range(num_heads)])

    def transformed_features(self, features):
        own, messages = {key: [] for key in ("P", "A", "state")}, {}
        for head in self.heads:
            own["P"].append(head.fc["P"](features["P"]))
            own["A"].append(head.fc["A"](features["A"]))
            own["state"].append(head.fc["in"](features["state"]))
            payloads = None if self.use_real else head.payloads(features)
            for source, name, target in RELATIONS:
                value = (head.fc[name](features[source])
                         if self.use_real or target == "state"
                         else head.decoder[target](payloads[source]))
                messages.setdefault(name, []).append(value)
        return ({key: torch.stack(values, dim=1) for key, values in own.items()},
                {key: torch.stack(values, dim=1) for key, values in messages.items()})

    def forward(self, graph, features, topology=None):
        own, messages = self.transformed_features(features)
        if self.message_backend == "dgl":
            if graph is None:
                raise ValueError("DGL paper-v1 forward requires a graph")
            received = self._dgl(graph, own, messages)
        else:
            if topology is None:
                raise ValueError("torch-v1 forward requires static topology")
            received = {}
            for _, name, target in RELATIONS:
                if topology.nonempty[name]:
                    received[name] = aggregate(messages[name], own[target],
                        self._attention(name, "src"), self._attention(name, "dst"),
                        topology.masks[name])
        outputs = {}
        for target, incoming in (("P", ("p2p", "a2p")),
                                 ("A", ("p2a", "a2a")),
                                 ("state", ("p2s", "a2s"))):
            value = own[target]
            for name in incoming:
                if name in received:
                    value = value + received[name]
            if self.merge == "cat":
                outputs[target] = F.relu(value).flatten(1)
            else:
                outputs[target] = value.mean(dim=1)
        return outputs

    def _attention(self, relation, side):
        return torch.stack([head.attention[relation + "_" + side][0]
                            for head in self.heads], dim=0)

    def _dgl(self, graph, own, messages):
        # Import only on the reference path. local_scope prevents learned tensors
        # from surviving in a supplied graph after this forward returns.
        import dgl.function as fn
        from dgl.ops import edge_softmax
        received = {}
        with graph.local_scope():
            for _, name, target in RELATIONS:
                relation = graph[name]
                if not relation.num_edges():
                    continue
                with relation.local_scope():
                    relation.srcdata["paper_message"] = messages[name]
                    relation.srcdata["paper_source"] = (
                        messages[name] * self._attention(name, "src")).sum(-1, keepdim=True)
                    relation.dstdata["paper_target"] = (
                        own[target] * self._attention(name, "dst")).sum(-1, keepdim=True)
                    relation.apply_edges(fn.u_add_v("paper_source", "paper_target", "paper_score"))
                    scores = F.leaky_relu(relation.edata["paper_score"], negative_slope=0.2)
                    relation.edata["paper_weight"] = edge_softmax(relation, scores)
                    relation.update_all(fn.u_mul_e("paper_message", "paper_weight", "paper_m"),
                                        fn.sum("paper_m", "paper_result"))
                    received[name] = relation.dstdata["paper_result"]
        return received
