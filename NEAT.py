# Neural Evolution of Augmenting Topologies (NEAT)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math
import random


@dataclass
class NodeGene:
    id: int
    type: str  # 'input', 'output', 'hidden', 'bias'


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


class InnovationTracker:
    def __init__(self):
        self.current_innovation = 0
        # key: (in_node, out_node) -> innovation number
        self.connection_innovations: Dict[Tuple[int, int], int] = {}

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.connection_innovations:
            self.current_innovation += 1
            self.connection_innovations[key] = self.current_innovation
        return self.connection_innovations[key]


class Genome:
    def __init__(self):
        self.nodes: Dict[int, NodeGene] = {}
        # innovation -> connection
        self.connections: Dict[int, ConnectionGene] = {}

    def copy(self) -> Genome:
        g = Genome()
        g.nodes = {nid: NodeGene(n.id, n.type) for nid, n in self.nodes.items()}
        g.connections = {inn: ConnectionGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation)
                         for inn, c in self.connections.items()}
        return g

    def add_node(self, node: NodeGene) -> None:
        self.nodes[node.id] = node

    def add_connection(self, conn: ConnectionGene) -> None:
        self.connections[conn.innovation] = conn


class Network:
    def __init__(self, input_layer: int, output_layer: int, seed: Optional[int] = None, device: Optional[str] = None, use_gpu_if_available: bool = True):
        if seed is not None:
            random.seed(seed)

        if input_layer <= 0 or output_layer <= 0:
            raise ValueError("input_layer and output_layer must be positive")

        self.input_size = input_layer
        self.output_size = output_layer

        # gene structures
        self.genome = Genome()
        self.innovation_tracker = InnovationTracker()

        # node id allocation: reserve 0 as bias
        self.next_node_id = 0

        # create bias node
        bias_id = self._allocate_node_id()
        self.genome.add_node(NodeGene(bias_id, 'bias'))

        # create input nodes
        self.input_ids: List[int] = []
        for _ in range(input_layer):
            nid = self._allocate_node_id()
            self.genome.add_node(NodeGene(nid, 'input'))
            self.input_ids.append(nid)

        # create output nodes
        self.output_ids: List[int] = []
        for _ in range(output_layer):
            nid = self._allocate_node_id()
            self.genome.add_node(NodeGene(nid, 'output'))
            self.output_ids.append(nid)

        # fully connect inputs and bias to outputs with random weights
        for in_id in [bias_id] + self.input_ids:
            for out_id in self.output_ids:
                    inn = self.innovation_tracker.get_connection_innovation(in_id, out_id)
                    w = random.uniform(-1.0, 1.0)
                    self.genome.add_connection(ConnectionGene(in_id, out_id, w, True, inn))

        # cache for topological order
        self._cached_topo: Optional[List[int]] = None

        # device / torch-related cache (lazy; only if torch is available)
        self.device: str = 'cpu'
        if device is not None:
            self.device = device
        elif use_gpu_if_available:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self.device = 'cuda'
            except Exception:
                # torch not available or cuda not available; stay on cpu
                self.device = 'cpu'

        # Lazy-built tensors for GPU forward pass
        self._torch_cache_built: bool = False
        self._torch_A = None  # adjacency sparse matrix (N x N)
        self._torch_layers: Optional[List[List[int]]] = None  # list of node idx layers (excluding inputs/bias)
        self._id_to_idx: Optional[Dict[int, int]] = None
        self._idx_bias: Optional[int] = None
        self._idx_inputs: Optional[List[int]] = None
        self._idx_outputs: Optional[List[int]] = None

        # Recurrent evaluation settings
        # When True, mutations may create cycles and evaluation runs for multiple steps
        self.recurrent = True
        # Number of recurrent update steps per activate() call
        self.activation_steps = 3

    # ---------- basic helpers ----------
    def _allocate_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _invalidate_caches(self) -> None:
        # Invalidate CPU topo cache and any GPU caches
        self._cached_topo = None
        self._torch_cache_built = False
        self._torch_A = None
        self._torch_layers = None
        self._id_to_idx = None
        self._idx_bias = None
        self._idx_inputs = None
        self._idx_outputs = None

    def clone(self) -> Network:
        child = Network(self.input_size, self.output_size, device=self.device)
        child.genome = self.genome.copy()
        child.innovation_tracker = self.innovation_tracker  # shared tracker reference
        child.next_node_id = self.next_node_id
        child.input_ids = list(self.input_ids)
        child.output_ids = list(self.output_ids)
        child._cached_topo = list(self._cached_topo) if self._cached_topo is not None else None
        # do not copy torch cache; rebuild lazily in child when needed
        child._torch_cache_built = False
        # copy recurrent settings
        child.recurrent = self.recurrent
        child.activation_steps = self.activation_steps
        return child

    # ---------- evaluation ----------
    def activate(self, inputs: List[float]) -> List[float]:
        # If configured for GPU and torch is available, try the accelerated path
        if self.device != 'cpu':
            try:
                outputs = self.activate_gpu(inputs)
                # activate_gpu returns a torch.Tensor or Python list; normalize to List[float]
                try:
                    import torch  # type: ignore
                    if isinstance(outputs, torch.Tensor):
                        # single example is (1, O) or (O,); convert to list
                        if outputs.dim() == 2 and outputs.size(0) == 1:
                            return outputs[0].detach().cpu().tolist()
                        if outputs.dim() == 1:
                            return outputs.detach().cpu().tolist()
                except Exception:
                    pass
                # If it is already a list, return directly
                if isinstance(outputs, list):
                    return outputs
            except Exception:
                # Fallback to CPU path below on any error
                pass
        if len(inputs) != self.input_size:
            raise ValueError("inputs length must match input_size")

        # set bias and inputs
        bias_id = None
        for nid, node in self.genome.nodes.items():
            if node.type == 'bias':
                bias_id = nid
                break
        if bias_id is None:
            raise RuntimeError("Bias node missing")

        # incoming adjacency by out_node
        incoming: Dict[int, List[ConnectionGene]] = {}
        for conn in self.genome.connections.values():
            if not conn.enabled:
                continue
            incoming.setdefault(conn.out_node, []).append(conn)

        if self.recurrent:
            # Recurrent update: iterate activation_steps times, clamping inputs and bias each step
            values: Dict[int, float] = {nid: 0.0 for nid in self.genome.nodes.keys()}
            for _ in range(self.activation_steps):
                # clamp bias and inputs
                values[bias_id] = 1.0
                for idx, nid in enumerate(self.input_ids):
                    values[nid] = inputs[idx]
                # compute new activations for non-input/bias nodes based on previous values
                new_values = dict(values)
                for nid, node in self.genome.nodes.items():
                    if node.type in ('input', 'bias'):
                        continue
                    total = 0.0
                    for conn in incoming.get(nid, []):
                        src_val = values.get(conn.in_node, 0.0)
                        total += conn.weight * src_val
                    new_values[nid] = self._activation_tanh(total)
                values = new_values
            return [values.get(oid, 0.0) for oid in self.output_ids]
        else:
            # Acyclic fast path using topological order
            order = self._get_topological_order()
            values: Dict[int, float] = {}
            values[bias_id] = 1.0
            for idx, nid in enumerate(self.input_ids):
                values[nid] = inputs[idx]
            for nid in order:
                node = self.genome.nodes[nid]
                if node.type in ('input', 'bias'):
                    continue
                total = 0.0
                for conn in incoming.get(nid, []):
                    src_val = values.get(conn.in_node, 0.0)
                    total += conn.weight * src_val
                values[nid] = self._activation_tanh(total)
            return [values.get(oid, 0.0) for oid in self.output_ids]

    def _activation_tanh(self, x: float) -> float:
        return math.tanh(x)

    def _get_topological_order(self) -> List[int]:
        if self._cached_topo is not None:
            return self._cached_topo

        # build graph
        out_edges: Dict[int, List[int]] = {}
        in_degree: Dict[int, int] = {nid: 0 for nid in self.genome.nodes}
        for conn in self.genome.connections.values():
            if not conn.enabled:
                continue
            out_edges.setdefault(conn.in_node, []).append(conn.out_node)
            in_degree[conn.out_node] = in_degree.get(conn.out_node, 0) + 1

        # Kahn's algorithm
        queue: List[int] = [nid for nid, node in self.genome.nodes.items() if node.type in ('input', 'bias')]
        seen: Set[int] = set(queue)
        order: List[int] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for m in out_edges.get(nid, []):
                in_degree[m] -= 1
                if in_degree[m] == 0 and m not in seen:
                    seen.add(m)
                    queue.append(m)

        # include any remaining nodes (isolated outputs)
        for nid in self.genome.nodes:
            if nid not in seen:
                order.append(nid)

        self._cached_topo = order
        return order

    # ---------- mutation operations ----------
    def mutate(self):
        if random.random() < 0.9:
            self.mutate_add_node()
        if random.random() < 0.8:
            self.mutate_add_connection()
        if random.random() < 0.7:
            self.mutate_weights()
        if random.random() < 0.6:
            count = random.randint(1, 10)
            for _ in range(count):
                self.mutate_delete_connection()
            
    
    def mutate_weights(self, perturb_chance: float = 0.9, sigma: float = 0.5) -> None:
        for conn in self.genome.connections.values():
            if random.random() < perturb_chance:
                conn.weight += random.gauss(0.0, sigma)
            else:
                conn.weight = random.uniform(-1.0, 1.0)
        self._invalidate_caches()

    def _creates_cycle(self, src: int, dst: int) -> bool:
        # DFS from dst to see if we can reach src
        stack = [dst]
        visited: Set[int] = set()
        adj: Dict[int, List[int]] = {}
        for c in self.genome.connections.values():
            if not c.enabled:
                continue
            adj.setdefault(c.in_node, []).append(c.out_node)
        while stack:
            n = stack.pop()
            if n == src:
                return True
            if n in visited:
                continue
            visited.add(n)
            stack.extend(adj.get(n, []))
        return False

    def mutate_add_connection(self, max_tries: int = 32) -> bool:
        node_ids = list(self.genome.nodes.keys())
        for _ in range(max_tries):
            a, b = random.sample(node_ids, 2)
            a_type = self.genome.nodes[a].type
            b_type = self.genome.nodes[b].type

            # enforce direction: no edges into inputs/bias
            if b_type in ('input', 'bias'):
                continue
            # In non-recurrent mode, disallow edges out of outputs and self-loops
            if not self.recurrent:
                if a_type == 'output':
                    continue
                if a == b:
                    continue

            # avoid duplicate
            exists = False
            for c in self.genome.connections.values():
                if c.in_node == a and c.out_node == b:
                    exists = True
                    break
            if exists:
                continue

            # avoid cycles only when not in recurrent mode
            if not self.recurrent and self._creates_cycle(a, b):
                continue

            inn = self.innovation_tracker.get_connection_innovation(a, b)
            w = random.uniform(-1.0, 1.0)
            self.genome.add_connection(ConnectionGene(a, b, w, True, inn))
            self._invalidate_caches()
            return True
        return False

    def mutate_delete_connection(self) -> bool:
        
        # Delete random connection
        enabled_conns = [c for c in self.genome.connections.values() if c.enabled]
        if not enabled_conns:
            return False
        conn = random.choice(enabled_conns)
        conn.enabled = False
        self._invalidate_caches()
        return True
    
    def mutate_add_node(self) -> bool:
        enabled_conns = [c for c in self.genome.connections.values() if c.enabled]
        if not enabled_conns:
            return False
        conn = random.choice(enabled_conns)
        conn.enabled = False

        # create new node
        new_node_id = self._allocate_node_id()
        self.genome.add_node(NodeGene(new_node_id, 'hidden'))

        # connect in -> new
        inn1 = self.innovation_tracker.get_connection_innovation(conn.in_node, new_node_id)
        self.genome.add_connection(ConnectionGene(conn.in_node, new_node_id, 1.0, True, inn1))

        # connect new -> out
        inn2 = self.innovation_tracker.get_connection_innovation(new_node_id, conn.out_node)
        self.genome.add_connection(ConnectionGene(new_node_id, conn.out_node, conn.weight, True, inn2))

        self._invalidate_caches()
        return True

    # ---------- crossover and distance ----------
    @staticmethod
    def crossover(parent1: Network, parent2: Network, fitness1: float, fitness2: float, rng_seed: Optional[int] = None) -> Network:
        if rng_seed is not None:
            random.seed(rng_seed)

        # assume parent1 is more fit or equal for simplicity
        if fitness2 > fitness1:
            parent1, parent2 = parent2, parent1
            fitness1, fitness2 = fitness2, fitness1

        child = Network(parent1.input_size, parent1.output_size)
        child.innovation_tracker = parent1.innovation_tracker
        child.next_node_id = max(parent1.next_node_id, parent2.next_node_id)

        # inherit nodes (union)
        for nid, node in parent1.genome.nodes.items():
            child.genome.add_node(NodeGene(nid, node.type))
        for nid, node in parent2.genome.nodes.items():
            if nid not in child.genome.nodes:
                child.genome.add_node(NodeGene(nid, node.type))

        # align connections by innovation
        conns1 = parent1.genome.connections
        conns2 = parent2.genome.connections
        all_innovs = set(conns1.keys()) | set(conns2.keys())
        for inn in all_innovs:
            c1 = conns1.get(inn)
            c2 = conns2.get(inn)
            chosen: Optional[ConnectionGene] = None
            if c1 and c2:
                chosen = random.choice([c1, c2])
            elif c1 and not c2:
                chosen = c1  # excess/disjoint from more fit parent
            elif c2 and not c1:
                # only inherit from less fit parent if matching NEAT precisely would exclude; here keep simple: skip
                continue
            if chosen:
                child.genome.add_connection(ConnectionGene(chosen.in_node, chosen.out_node, chosen.weight, chosen.enabled, chosen.innovation))

        child.input_ids = list(parent1.input_ids)
        child.output_ids = list(parent1.output_ids)
        # inherit recurrent settings from the more-fit parent
        child.recurrent = parent1.recurrent
        child.activation_steps = parent1.activation_steps
        child._invalidate_caches()
        return child

    # ---------- GPU acceleration (optional; requires torch) ----------
    def _build_torch_cache(self) -> None:
        if self._torch_cache_built:
            return
        # Build only if torch is present
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Torch is required for GPU acceleration but is not available") from e

        order = self._get_topological_order()

        # For recurrent mode we still build indices and adjacency, but layer info is unused
        # Compute predecessors and layer assignment (used only in non-recurrent GPU path)
        predecessors: Dict[int, List[int]] = {}
        for c in self.genome.connections.values():
            if not c.enabled:
                continue
            predecessors.setdefault(c.out_node, []).append(c.in_node)

        layer_of: Dict[int, int] = {}
        for nid, node in self.genome.nodes.items():
            if node.type in ('input', 'bias'):
                layer_of[nid] = 0
        for nid in order:
            if nid not in layer_of:
                preds = predecessors.get(nid, [])
                if preds:
                    layer_of[nid] = 1 + max(layer_of.get(p, 0) for p in preds)
                else:
                    layer_of[nid] = 0

        # Ensure outputs at far right
        max_layer = max(layer_of.values()) if layer_of else 0
        for oid in self.output_ids:
            max_layer = max(max_layer, layer_of.get(oid, 0))
        for oid in self.output_ids:
            layer_of[oid] = max_layer

        # Build stable index mapping following topological order
        id_to_idx: Dict[int, int] = {nid: idx for idx, nid in enumerate(order)}

        # Identify special indices
        bias_id = None
        for nid, node in self.genome.nodes.items():
            if node.type == 'bias':
                bias_id = nid
                break
        if bias_id is None:
            raise RuntimeError("Bias node missing")

        idx_bias = id_to_idx[bias_id]
        idx_inputs = [id_to_idx[nid] for nid in self.input_ids]
        idx_outputs = [id_to_idx[nid] for nid in self.output_ids]

        # Build sparse adjacency A (rows: target, cols: source)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for conn in self.genome.connections.values():
            if not conn.enabled:
                continue
            rows.append(id_to_idx[conn.out_node])
            cols.append(id_to_idx[conn.in_node])
            vals.append(conn.weight)

        num_nodes = len(order)
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
            A = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
            A = A.coalesce().to(self.device)
        else:
            # No connections -> zero sparse matrix
            A = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32), size=(num_nodes, num_nodes)).to(self.device)

        # Group nodes by layer (used only if non-recurrent GPU path)
        layers: Dict[int, List[int]] = {}
        for nid in order:
            layers.setdefault(layer_of[nid], []).append(id_to_idx[nid])
        ordered_layers: List[List[int]] = []
        if not self.recurrent:
            for ly in range(1, max_layer + 1):
                if ly in layers:
                    ordered_layers.append(layers[ly])

        self._torch_A = A
        self._torch_layers = ordered_layers
        self._id_to_idx = id_to_idx
        self._idx_bias = idx_bias
        self._idx_inputs = idx_inputs
        self._idx_outputs = idx_outputs
        self._torch_cache_built = True

    def activate_gpu(self, inputs):
        """GPU-accelerated forward pass using torch sparse ops.

        Accepts a single input (shape [I]) or a batch (shape [B, I]) and returns
        a torch.Tensor on the configured device.
        """
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Torch is required for GPU acceleration but is not available") from e

        if self.device == 'cpu':
            # If not using GPU, still allow torch path on CPU
            pass

        self._build_torch_cache()

        # Normalize inputs to tensor with shape (B, I)
        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self.device, dtype=torch.float32)
        else:
            x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.input_size:
            raise ValueError("inputs length must match input_size")

        num_nodes = self._torch_A.size(0)  # type: ignore
        B = x.size(0)

        # values: (B, N)
        values = torch.zeros((B, num_nodes), dtype=torch.float32, device=self.device)
        # Set bias and inputs
        values[:, self._idx_bias] = 1.0  # type: ignore
        values[:, self._idx_inputs] = x  # type: ignore

        if self.recurrent:
            # Recurrent iterative updates with clamped inputs and bias
            for _ in range(self.activation_steps):
                y = torch.sparse.mm(self._torch_A, values.transpose(0, 1))  # (N, B)
                preact = y.transpose(0, 1)  # (B, N)
                values = torch.tanh(preact)
                # clamp bias and inputs each step
                values[:, self._idx_bias] = 1.0  # type: ignore
                values[:, self._idx_inputs] = x  # type: ignore
        else:
            # Layer-wise propagation for acyclic networks
            if self._torch_layers:
                for layer_nodes in self._torch_layers:
                    y = torch.sparse.mm(self._torch_A, values.transpose(0, 1))  # type: ignore
                    layer_idx = torch.tensor(layer_nodes, dtype=torch.long, device=self.device)
                    preact = y.index_select(0, layer_idx).transpose(0, 1)
                    activated = torch.tanh(preact)
                    values.index_copy_(1, layer_idx, activated)

        # Gather outputs -> (B, O)
        out_idx = torch.tensor(self._idx_outputs, dtype=torch.long, device=self.device)  # type: ignore
        outputs = values.index_select(1, out_idx)

        # If single example, return 1D tensor of size (O,)
        if outputs.size(0) == 1:
            return outputs.squeeze(0)
        return outputs

    @staticmethod
    def distance(g1: Network, g2: Network, c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        conns1 = g1.genome.connections
        conns2 = g2.genome.connections
        innovs1 = sorted(conns1.keys())
        innovs2 = sorted(conns2.keys())

        i = j = 0
        disjoint = 0
        excess = 0
        weight_diffs: List[float] = []

        max_inn1 = innovs1[-1] if innovs1 else -1
        max_inn2 = innovs2[-1] if innovs2 else -1

        while i < len(innovs1) and j < len(innovs2):
            inn1 = innovs1[i]
            inn2 = innovs2[j]
            if inn1 == inn2:
                w1 = conns1[inn1].weight
                w2 = conns2[inn2].weight
                weight_diffs.append(abs(w1 - w2))
                i += 1
                j += 1
            elif inn1 < inn2:
                if inn1 > max_inn2:
                    excess += 1
                else:
                    disjoint += 1
                i += 1
            else:
                if inn2 > max_inn1:
                    excess += 1
                else:
                    disjoint += 1
                j += 1

        # remaining are excess
        excess += (len(innovs1) - i) + (len(innovs2) - j)

        N = max(len(innovs1), len(innovs2))
        if N < 20:
            N = 1  # per NEAT normalization heuristic
        W = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0.0
        return c1 * excess / N + c2 * disjoint / N + c3 * W



    # ---------- visualization ----------
    def render(self, surface, topleft: Tuple[int, int] = (0, 0), size: Tuple[int, int] = (400, 300)) -> None:
        """Render the network onto a pygame surface.

        - Inputs and bias are placed on the left, outputs on the right, hidden by inferred layers.
        - Connection color encodes weight sign (green positive, red negative), thickness encodes |weight|.
        - Disabled connections are drawn in gray.

        Parameters
        ----------
        surface: pygame.Surface
            Target surface to draw onto.
        topleft: (int, int)
            Top-left coordinate of the drawing area.
        size: (int, int)
            Size (width, height) of the drawing area.
        """
        # Local import to avoid hard dependency when not visualizing
        import pygame  # type: ignore

        x0, y0 = topleft
        width, height = size

        # Padding inside the drawing rect
        pad_x = max(12, width // 20)
        pad_y = max(12, height // 20)

        draw_w = max(1, width - 2 * pad_x)
        draw_h = max(1, height - 2 * pad_y)

        # Determine layers using topological order and predecessors
        order = self._get_topological_order()

        # Build predecessors map (enabled connections only)
        predecessors: Dict[int, List[int]] = {}
        for c in self.genome.connections.values():
            if not c.enabled:
                continue
            predecessors.setdefault(c.out_node, []).append(c.in_node)

        # Initialize layers for all nodes
        layer_of: Dict[int, int] = {}
        for nid, node in self.genome.nodes.items():
            if node.type in ('input', 'bias'):
                layer_of[nid] = 0

        # Propagate layers in topological order
        for nid in order:
            if nid not in layer_of:
                preds = predecessors.get(nid, [])
                if preds:
                    layer_of[nid] = 1 + max(layer_of.get(p, 0) for p in preds)
                else:
                    layer_of[nid] = 0

        # Ensure outputs are at the far right layer
        max_layer = max(layer_of.values()) if layer_of else 0
        for oid in self.output_ids:
            max_layer = max(max_layer, layer_of.get(oid, 0))
        for oid in self.output_ids:
            layer_of[oid] = max_layer

        # Group nodes by layer (sorted for stable layout)
        nodes_by_layer: Dict[int, List[int]] = {}
        for nid in order:
            nodes_by_layer.setdefault(layer_of[nid], []).append(nid)
        for ly in nodes_by_layer:
            nodes_by_layer[ly].sort()

        num_layers = max_layer + 1
        # Avoid division by zero for single-layer nets
        def layer_to_x(layer_idx: int) -> int:
            if num_layers <= 1:
                return x0 + pad_x + draw_w // 2
            return x0 + pad_x + int(round(layer_idx * (draw_w / (num_layers - 1))))

        # Compute node positions
        node_pos: Dict[int, Tuple[int, int]] = {}
        for ly in range(num_layers):
            nodes_in_layer = nodes_by_layer.get(ly, [])
            count = len(nodes_in_layer)
            if count == 0:
                continue
            # Evenly space nodes vertically within the drawing area
            step = draw_h / (count + 1)
            for idx, nid in enumerate(nodes_in_layer, start=1):
                x = layer_to_x(ly)
                y = y0 + pad_y + int(round(idx * step))
                node_pos[nid] = (x, y)

        # Drawing parameters
        node_radius = max(4, min(14, min(width, height) // 30))

        def color_for_node(node_type: str) -> Tuple[int, int, int]:
            if node_type == 'input':
                return (66, 135, 245)  # blue
            if node_type == 'output':
                return (245, 166, 35)  # orange
            if node_type == 'bias':
                return (230, 230, 50)  # yellow
            return (240, 240, 240)     # light

        def color_for_weight(weight: float, enabled: bool) -> Tuple[int, int, int]:
            if not enabled:
                return (120, 120, 120)
            if weight >= 0:
                g = 80 + int(175 * min(1.0, abs(weight)))
                return (40, g, 40)
            r = 80 + int(175 * min(1.0, abs(weight)))
            return (r, 40, 40)

        def thickness_for_weight(weight: float) -> int:
            return max(1, int(1 + 2 * min(1.0, abs(weight))))

        # Draw connections first
        for conn in self.genome.connections.values():
            p1 = node_pos.get(conn.in_node)
            p2 = node_pos.get(conn.out_node)
            if not p1 or not p2:
                continue
            color = color_for_weight(conn.weight, conn.enabled)
            width_px = thickness_for_weight(conn.weight)
            pygame.draw.line(surface, color, p1, p2, width_px)

        # Draw nodes on top
        for nid, node in self.genome.nodes.items():
            pos = node_pos.get(nid)
            if not pos:
                continue
            pygame.draw.circle(surface, color_for_node(node.type), pos, node_radius)

        # Optional: outline the drawing rect lightly for context
        rect_color = (80, 80, 80)
        pygame.draw.rect(surface, rect_color, pygame.Rect(x0, y0, width, height), 1)