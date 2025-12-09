import simpy
import numpy as np
import random
import matplotlib.pyplot as plt

# ====================================
# Model parameters
# ====================================
NUM_CARS_DEFAULT = 100
PRESSURE_PERIOD = 1.0
SIM_TIME = 600.0

# Car geometry
CAR_LENGTH_MIN = 20.0
CAR_LENGTH_MAX = 40.0

# Radio physics - MATCHED TO NON-CLUSTERING CODE
MAX_LORA_RANGE = 900.0           # Max reliable LoRa distance [m]
CLUSTER_RADIUS = 600.0           # Realistic cluster coverage
RELIABILITY_THRESHOLD = 0.9      # Min per-link success probability
CHANNEL_DEGRADATION = 1        # Scale factor (0-1) to degrade all links
INTERFERENCE_DROP_PROB = 0.0     # Extra drop probability from other trains

# LoRa airtimes (realistic)
BACKBONE_AIRTIME = 0.030         # 30 ms per cluster-head packet
MEMBER_AIRTIME   = 0.003         # 3 ms per member short-hop packet

VERBOSE = False

# ====================================
# Packets
# ====================================
class Packet:
    _next_id = 0
    def __init__(self, origin_id, created_time):
        self.id = Packet._next_id
        Packet._next_id += 1
        self.origin_id = origin_id
        self.created_time = created_time
        self.hops = 0

class AggregatePacket:
    def __init__(self, cluster_id, created_time, sensor_data):
        self.cluster_id = cluster_id
        self.created_time = created_time
        self.sensor_data = sensor_data
        self.hops = 0


# ====================================
# Node
# ====================================
class Node:
    def __init__(self, env, node_id, pos, net, is_hot=False):
        self.env = env
        self.id = node_id
        self.pos = pos
        self.net = net
        self.is_hot = is_hot

        self.own_packets = []        # member → head
        self.to_head_packets = []    # member → head collection
        self.upstream_packets = []   # head → backbone

        self.last_agg_time = 0.0
        
        # Statistics tracking
        self.packets_sent = 0
        self.packets_dropped = 0
        self.queue_samples = []
        self.queue_times = []

        if not self.is_hot:
            self.env.process(self.generate_sensor_data())

    # ------------------------------------
    def generate_sensor_data(self):
        while True:
            yield self.env.timeout(PRESSURE_PERIOD)
            p = Packet(self.id, self.env.now)
            self.own_packets.append(p)

    def sample_queue(self):
        """Sample queue length for statistics"""
        total = len(self.own_packets) + len(self.to_head_packets) + len(self.upstream_packets)
        self.queue_samples.append(total)
        self.queue_times.append(self.env.now)

    # ------------------------------------
    # Member TDMA (intra-cluster)
    # ------------------------------------
    def send_member_slot(self):
        self.sample_queue()
        
        cid = self.net.cluster_id[self.id]
        head = self.net.cluster_head[cid]

        if self.id == head:
            return

        if not self.own_packets:
            return

        pkt = self.own_packets.pop(0)

        # PROBABILISTIC SHORT-HOP: Apply channel model
        d = abs(self.pos - self.net.positions[head])
        p_success = self.net.link_success_prob(d)
        p_link = p_success * CHANNEL_DEGRADATION * (1.0 - INTERFERENCE_DROP_PROB)
        p_link = max(0.0, min(1.0, p_link))
        
        success = (random.random() < p_link)
        
        # Record transmission attempt
        self.net.record_transmission(self.id, head, d, p_link, success, is_backbone=False)
        
        if VERBOSE:
            print(f"[{self.env.now:8.3f}s] Member {self.id:3d} -> Head {head:3d} "
                  f"dist={d:6.1f}m Psucc={p_link:4.2f} "
                  f"{'OK' if success else 'DROP'} pkt {pkt.id}")
        
        if success:
            pkt.hops += 1
            self.net.nodes[head].to_head_packets.append(pkt)
            self.packets_sent += 1
        else:
            self.packets_dropped += 1

    # ------------------------------------
    # Backbone TDMA (cluster head forwarding)
    # ------------------------------------
    def send_backbone_slot(self):
        self.sample_queue()
        
        cid = self.net.cluster_id[self.id]
        head = self.net.cluster_head[cid]
        if self.id != head:
            return

        # Generate 1 aggregate packet every 1 second per cluster head
        if (self.env.now - self.last_agg_time) >= 1.0:
            agg = self.net.create_aggregate_packet(cid, self.env.now)
            self.upstream_packets.append(agg)
            self.last_agg_time = self.env.now

        if not self.upstream_packets:
            return

        pkt = self.upstream_packets.pop(0)

        nh = self.net.next_cluster_head(cid)
        if nh is None:   # deliver to HOT
            self.net.record_delivery(pkt)
            return

        # PROBABILISTIC LONG-HOP: Apply channel model
        d = abs(self.pos - self.net.positions[nh])
        p_success = self.net.link_success_prob(d)
        p_link = p_success * CHANNEL_DEGRADATION * (1.0 - INTERFERENCE_DROP_PROB)
        p_link = max(0.0, min(1.0, p_link))
        
        success = (random.random() < p_link)
        
        # Record transmission attempt
        self.net.record_transmission(self.id, nh, d, p_link, success, is_backbone=True)
        
        if VERBOSE:
            print(f"[{self.env.now:8.3f}s] Head {self.id:3d} -> Head {nh:3d} "
                  f"dist={d:6.1f}m Psucc={p_link:4.2f} "
                  f"{'OK' if success else 'DROP'} agg {pkt.cluster_id}")
        
        if success:
            pkt.hops += 1
            self.net.nodes[nh].upstream_packets.append(pkt)
            self.packets_sent += 1
        else:
            self.packets_dropped += 1


# ====================================
# Train Network
# ====================================
class TrainNetwork:
    def __init__(self, env, n):
        self.env = env
        self.N = n

        self.positions = self._generate_positions(n)
        self.hot_id = n - 1

        self.nodes = [
            Node(env, i, self.positions[i], self, is_hot=(i==self.hot_id))
            for i in range(n)
        ]

        # Realistic clustering
        self.cluster_id = self.form_clusters_realistic()
        self.clusters = self.build_cluster_lists()
        self.cluster_head = self.elect_cluster_heads()

        # Stats
        self.latencies = []
        self.hops_record = []
        self.latency_by_origin = {i: [] for i in range(n)}
        
        # Transmission statistics
        self.transmissions = {
            'distances': [],
            'probabilities': [],
            'successes': [],
            'is_backbone': []
        }
        
        # Energy efficiency metrics
        self.energy_stats = {
            'total_tx_time': 0.0,           # Total air time (successful + failed)
            'successful_tx_time': 0.0,      # Only successful transmissions
            'member_tx_time': 0.0,          # Member tier transmissions
            'backbone_tx_time': 0.0,        # Backbone tier transmissions
            'tx_count': 0,                  # Total transmission attempts
            'successful_tx_count': 0,       # Successful transmissions
            'per_node_tx_time': [0.0] * n, # Time each node spent transmitting
            'per_node_tx_count': [0] * n   # Number of transmissions per node
        }

        # TDMA processes
        self.env.process(self.backbone_tdma())
        self.env.process(self.member_tdma())

    # ------------------------------------
    def _generate_positions(self, n):
        lengths = np.random.uniform(CAR_LENGTH_MIN, CAR_LENGTH_MAX, size=n)
        pos = np.zeros(n)
        for i in range(1, n):
            pos[i] = pos[i-1] + lengths[i-1]
        return pos

    # ------------------------------------
    # PROBABILISTIC LINK MODEL - MATCHED TO NON-CLUSTERING CODE
    # ------------------------------------
    def link_success_prob(self, distance):
        """Packet success probability vs distance (exponential decay)."""
        if distance <= 0 or distance > MAX_LORA_RANGE:
            return 0.0
        # Same decay model as non-clustering code
        decay_distance = 1500.0
        p = np.exp(-(distance / decay_distance) ** 2)
        return float(max(0.0, min(1.0, p)))

    # ------------------------------------
    def record_transmission(self, src, dst, distance, prob, success, is_backbone):
        """Record transmission statistics"""
        self.transmissions['distances'].append(distance)
        self.transmissions['probabilities'].append(prob)
        self.transmissions['successes'].append(1 if success else 0)
        self.transmissions['is_backbone'].append(is_backbone)
        
        # Energy metrics
        tx_time = BACKBONE_AIRTIME if is_backbone else MEMBER_AIRTIME
        self.energy_stats['total_tx_time'] += tx_time
        self.energy_stats['tx_count'] += 1
        self.energy_stats['per_node_tx_time'][src] += tx_time
        self.energy_stats['per_node_tx_count'][src] += 1
        
        if success:
            self.energy_stats['successful_tx_time'] += tx_time
            self.energy_stats['successful_tx_count'] += 1
        
        if is_backbone:
            self.energy_stats['backbone_tx_time'] += tx_time
        else:
            self.energy_stats['member_tx_time'] += tx_time

    # ------------------------------------
    # REALISTIC CLUSTERING BY DISTANCE
    # ------------------------------------
    def form_clusters_realistic(self):
        cid = np.zeros(self.N, dtype=int)
        current = 0
        cid[0] = current
        current_head_pos = self.positions[0]

        for i in range(1, self.N):
            d = abs(self.positions[i] - current_head_pos)
            if d <= CLUSTER_RADIUS:
                cid[i] = current
            else:
                current += 1
                cid[i] = current
                current_head_pos = self.positions[i]

        return cid

    def build_cluster_lists(self):
        clusters = {}
        for i, c in enumerate(self.cluster_id):
            clusters.setdefault(c, []).append(i)
        return clusters

    # ------------------------------------
    def elect_cluster_heads(self):
        heads = {}
        cluster_ids = sorted(self.clusters.keys())

        for cid in reversed(cluster_ids):
            members = self.clusters[cid]

            if cid == cluster_ids[-1]:
                target_id = self.hot_id
            else:
                target_id = heads[cid + 1]

            target_pos = self.positions[target_id]

            # Select member with best (most reliable) link to target
            best = None
            best_p = -1.0
            for m in members:
                d = abs(self.positions[m] - target_pos)
                if d > MAX_LORA_RANGE:
                    continue
                p = self.link_success_prob(d)
                if p > best_p:
                    best_p = p
                    best = m

            # Fallback to closest member if no good link found
            if best is None:
                best = min(members, key=lambda m: abs(self.positions[m] - target_pos))
            
            heads[cid] = best
        return heads

    # ------------------------------------
    def next_cluster_head(self, cid):
        if cid + 1 in self.cluster_head:
            return self.cluster_head[cid + 1]
        return None

    # ------------------------------------
    # Aggregation
    # ------------------------------------
    def create_aggregate_packet(self, cid, t):
        head = self.cluster_head[cid]
        data = {}

        # Collect all packets from cluster members, keeping only the most recent from each node
        for p in self.nodes[head].to_head_packets:
            # Only keep this packet if we don't have data from this node yet, 
            # or if this packet is newer than what we already have
            if p.origin_id not in data or p.created_time > data[p.origin_id]:
                data[p.origin_id] = p.created_time
        
        # Clear all collected member packets (discard old ones)
        self.nodes[head].to_head_packets = []

        # Add head's own most recent packet
        if self.nodes[head].own_packets:
            # Find the most recent packet from the head itself
            most_recent_idx = -1
            most_recent_time = -1
            for idx, p in enumerate(self.nodes[head].own_packets):
                if p.created_time > most_recent_time:
                    most_recent_time = p.created_time
                    most_recent_idx = idx
            
            # Add the most recent packet
            if most_recent_idx >= 0:
                p = self.nodes[head].own_packets[most_recent_idx]
                data[p.origin_id] = p.created_time
            
            # Clear all head's own packets (discard old ones)
            self.nodes[head].own_packets = []

        return AggregatePacket(cid, t, data)

    # ------------------------------------
    def record_delivery(self, agg):
        now = self.env.now
        for origin, t0 in agg.sensor_data.items():
            latency = now - t0
            self.latencies.append(latency)
            self.latency_by_origin[origin].append(latency)
        self.hops_record.append(agg.hops)

    # ------------------------------------
    # Tier 1 backbone TDMA
    # ------------------------------------
    def backbone_tdma(self):
        heads = sorted(self.cluster_head.values())
        H = len(heads)
        idx = 0
        while self.env.now < SIM_TIME:
            nid = heads[idx % H]
            self.nodes[nid].send_backbone_slot()
            idx += 1
            yield self.env.timeout(BACKBONE_AIRTIME)

    # ------------------------------------
    # Tier 2 member TDMA
    # ------------------------------------
    def member_tdma(self):
        members = [n for n in range(self.N) if n not in self.cluster_head.values()]
        M = len(members)
        if M == 0:
            return
        idx = 0
        while self.env.now < SIM_TIME:
            nid = members[idx % M]
            self.nodes[nid].send_member_slot()
            idx += 1
            yield self.env.timeout(MEMBER_AIRTIME)


# ====================================
# VISUALIZATION
# ====================================
def plot_comprehensive_analysis(results, net):
    """Create comprehensive visualization of system behavior in separate windows"""
    
    # Calculate common data
    distances = np.linspace(0, MAX_LORA_RANGE, 1000)
    probabilities = [net.link_success_prob(d) for d in distances]
    cluster_sizes = [len(members) for members in net.clusters.values()]
    trans = net.transmissions
    energy = net.energy_stats
    
    # 1. Signal Degradation Model
    plt.figure(figsize=(10, 6))
    plt.plot(distances, probabilities, 'b-', linewidth=2)
    plt.axhline(y=RELIABILITY_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({RELIABILITY_THRESHOLD})')
    plt.axvline(x=CLUSTER_RADIUS, color='g', linestyle='--', alpha=0.5, label=f'Cluster Radius ({CLUSTER_RADIUS}m)')
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title('LoRa Link Success vs Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 2. Train Layout with Clusters
    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(net.clusters)))
    for cid, members in net.clusters.items():
        positions = [net.positions[m] for m in members]
        plt.scatter(positions, [cid]*len(positions), c=[colors[cid]], s=50, alpha=0.6)
        head = net.cluster_head[cid]
        plt.scatter(net.positions[head], cid, c=[colors[cid]], s=200, marker='*', 
                   edgecolors='black', linewidths=2)
    plt.xlabel('Position along train (m)', fontsize=12)
    plt.ylabel('Cluster ID', fontsize=12)
    plt.title(f'Train Layout: {net.N} cars, {len(net.clusters)} clusters (★ = cluster head)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Latency Distribution
    if len(results["latencies"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(results["latencies"], bins=50, edgecolor="black", alpha=0.7, color='steelblue')
        plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='1s threshold')
        plt.xlabel("Latency (s)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f'Latency Distribution (Mean: {results["latencies"].mean():.3f}s)', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    # 4. Hop Count Distribution
    if len(results["hops"]) > 0:
        plt.figure(figsize=(10, 6))
        hop_counts = np.bincount(results["hops"].astype(int))
        plt.bar(range(len(hop_counts)), hop_counts, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel("Number of Hops", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f'Hop Distribution (Mean: {results["hops"].mean():.2f})', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 5. Transmission Success by Distance
    if len(trans['distances']) > 0:
        plt.figure(figsize=(12, 6))
        backbone_idx = [i for i, b in enumerate(trans['is_backbone']) if b]
        member_idx = [i for i, b in enumerate(trans['is_backbone']) if not b]
        
        if backbone_idx:
            bb_dist = [trans['distances'][i] for i in backbone_idx]
            bb_succ = [trans['successes'][i] for i in backbone_idx]
            plt.scatter(bb_dist, bb_succ, alpha=0.3, s=10, label='Backbone', c='red')
        
        if member_idx:
            mb_dist = [trans['distances'][i] for i in member_idx]
            mb_succ = [trans['successes'][i] for i in member_idx]
            plt.scatter(mb_dist, mb_succ, alpha=0.3, s=10, label='Member', c='blue')
        
        plt.xlabel('Distance (m)', fontsize=12)
        plt.ylabel('Success (1) / Failure (0)', fontsize=12)
        plt.title('Transmission Outcomes by Distance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    # 6. Success Rate vs Distance (binned)
    if len(trans['distances']) > 0:
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, MAX_LORA_RANGE, 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        success_rates = []
        for i in range(len(bins)-1):
            mask = [(bins[i] <= d < bins[i+1]) for d in trans['distances']]
            if sum(mask) > 0:
                rate = np.mean([trans['successes'][j] for j, m in enumerate(mask) if m])
                success_rates.append(rate)
            else:
                success_rates.append(np.nan)
        
        plt.plot(bin_centers, success_rates, 'o-', linewidth=2, markersize=8, label='Empirical')
        plt.plot(distances, probabilities, 'r--', alpha=0.5, linewidth=2, label='Theoretical')
        plt.xlabel('Distance (m)', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('Measured vs Theoretical Link Success', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    # 7. Latency by Origin Position
    plt.figure(figsize=(12, 6))
    origin_positions = []
    mean_latencies = []
    for origin_id, lats in net.latency_by_origin.items():
        if len(lats) > 0:
            origin_positions.append(net.positions[origin_id])
            mean_latencies.append(np.mean(lats))
    
    if origin_positions:
        plt.scatter(origin_positions, mean_latencies, s=80, alpha=0.6, c='purple')
        plt.xlabel('Node Position on Train (m)', fontsize=12)
        plt.ylabel('Mean Latency (s)', fontsize=12)
        plt.title('Latency vs Position on Train', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 8. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_sizes, bins=range(1, max(cluster_sizes)+2), 
             edgecolor='black', alpha=0.7, align='left', color='green')
    plt.xlabel('Cluster Size (nodes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Cluster Size Distribution (Mean: {np.mean(cluster_sizes):.1f} nodes)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 9. Delivery Probability over Time
    if len(results["latencies"]) > 0:
        plt.figure(figsize=(12, 6))
        delivery_times = np.array(results["latencies"]) + np.array([0]*len(results["latencies"]))
        time_bins = np.linspace(0, SIM_TIME, 50)
        deliveries_per_bin = np.histogram(delivery_times, bins=time_bins)[0]
        cumulative_deliveries = np.cumsum(deliveries_per_bin)
        expected_cumulative = time_bins[1:] * (net.N - 1)
        
        plt.plot(time_bins[1:], cumulative_deliveries / expected_cumulative, 
                linewidth=2, color='darkblue')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Cumulative Delivery Ratio', fontsize=12)
        plt.title('Delivery Performance Over Time', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3)
    
    # 10. Queue Length Statistics
    plt.figure(figsize=(12, 6))
    sample_nodes = [0, net.N//4, net.N//2, 3*net.N//4]
    for node_id in sample_nodes:
        if node_id < net.N and len(net.nodes[node_id].queue_samples) > 0:
            plt.plot(net.nodes[node_id].queue_times, 
                    net.nodes[node_id].queue_samples,
                    alpha=0.7, linewidth=1.5, label=f'Node {node_id}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Queue Length', fontsize=12)
    plt.title('Queue Evolution (Sample Nodes)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 11. Per-Node Delivery Success Rate
    plt.figure(figsize=(14, 6))
    node_delivery_counts = [len(net.latency_by_origin[i]) for i in range(net.N)]
    expected_per_node = SIM_TIME
    delivery_rates = [c / expected_per_node for c in node_delivery_counts]
    
    plt.bar(range(net.N), delivery_rates, alpha=0.7, color='teal')
    plt.axhline(y=results['p_delivery'], color='r', linestyle='--', linewidth=2,
                label=f'Overall: {results["p_delivery"]:.3f}')
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Delivery Rate', fontsize=12)
    plt.title('Per-Node Delivery Success', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 12. System Performance Summary
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Calculate energy efficiency metrics
    energy_per_delivered = energy['total_tx_time'] / results['delivered'] if results['delivered'] > 0 else 0
    energy_per_originated = energy['total_tx_time'] / results['originated'] if results['originated'] > 0 else 0
    tx_efficiency = energy['successful_tx_time'] / energy['total_tx_time'] if energy['total_tx_time'] > 0 else 0
    
    summary_text = f"""
    ═══════════════════════════════════════════════════════
                SYSTEM PERFORMANCE SUMMARY
    ═══════════════════════════════════════════════════════
    
    NETWORK CONFIGURATION:
    ─────────────────────────────────────────────────────── 
    • Total nodes:                {net.N}
    • Number of clusters:         {len(net.clusters)}
    • Average cluster size:       {np.mean(cluster_sizes):.1f} nodes
    • Train length:               {net.positions[-1]:.1f} m
    • Max LoRa range:             {MAX_LORA_RANGE} m
    • Cluster radius:             {CLUSTER_RADIUS} m
    
    PACKET STATISTICS:
    ───────────────────────────────────────────────────────
    • Packets originated:         {results['originated']:,}
    • Packets delivered:          {results['delivered']:,}
    • Overall delivery rate:      {results['p_delivery']:.1%}
    • Delivered within 1s:        {results['p_under_1s']:.1%}
    
    LATENCY (delivered packets only):
    ───────────────────────────────────────────────────────
    • Mean latency:               {results['latencies'].mean():.3f} s
    • Median latency:             {np.median(results['latencies']):.3f} s
    • 95th percentile:            {np.percentile(results['latencies'], 95):.3f} s
    • Max latency:                {results['latencies'].max():.3f} s
    
    HOP STATISTICS:
    ───────────────────────────────────────────────────────
    • Mean hops per packet:       {results['hops'].mean():.2f}
    • Median hops:                {np.median(results['hops']):.1f}
    • Max hops:                   {results['hops'].max():.0f}
    
    LINK PERFORMANCE:
    ───────────────────────────────────────────────────────
    • Total transmissions:        {len(trans['distances']):,}
    • Overall success rate:       {np.mean(trans['successes']):.1%}
    • Backbone transmissions:     {sum(trans['is_backbone']):,}
    • Member transmissions:       {len(trans['is_backbone']) - sum(trans['is_backbone']):,}
    
    ENERGY EFFICIENCY:
    ───────────────────────────────────────────────────────
    • Total TX time:              {energy['total_tx_time']:.2f} s
    • Successful TX time:         {energy['successful_tx_time']:.2f} s
    • Wasted TX time (drops):     {energy['total_tx_time'] - energy['successful_tx_time']:.2f} s
    • TX efficiency:              {tx_efficiency:.1%}
    • Backbone TX time:           {energy['backbone_tx_time']:.2f} s ({energy['backbone_tx_time']/energy['total_tx_time']*100:.1f}%)
    • Member TX time:             {energy['member_tx_time']:.2f} s ({energy['member_tx_time']/energy['total_tx_time']*100:.1f}%)
    • TX time per delivered pkt:  {energy_per_delivered:.4f} s
    • TX time per originated pkt: {energy_per_originated:.4f} s
    • Avg TX time per node:       {np.mean(energy['per_node_tx_time']):.3f} s
    
    ═══════════════════════════════════════════════════════
    """
    plt.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # 13. Per-Node Transmission Time (Energy Usage)
    plt.figure(figsize=(14, 6))
    node_colors = ['red' if i in net.cluster_head.values() else 'blue' for i in range(net.N)]
    bars = plt.bar(range(net.N), energy['per_node_tx_time'], alpha=0.7, color=node_colors)
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Total Transmission Time (s)', fontsize=12)
    plt.title('Energy Usage per Node (Red = Cluster Head, Blue = Member)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Cluster Head'),
                      Patch(facecolor='blue', alpha=0.7, label='Member')]
    plt.legend(handles=legend_elements, fontsize=10)
    
    # 14. Energy Efficiency Breakdown
    plt.figure(figsize=(12, 6))
    
    # Pie chart of TX time allocation
    plt.subplot(1, 2, 1)
    tx_allocation = [
        energy['successful_tx_time'],
        energy['total_tx_time'] - energy['successful_tx_time']
    ]
    labels = ['Successful TX', 'Wasted TX\n(Dropped)']
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(tx_allocation, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Transmission Time Efficiency', fontsize=12, fontweight='bold')
    
    # Bar chart of tier breakdown
    plt.subplot(1, 2, 2)
    tier_data = {
        'Node': energy['member_tx_time'],
        'Cluster': energy['backbone_tx_time']
    }
    bars = plt.bar(tier_data.keys(), tier_data.values(), color=['#3498db', '#e67e22'], alpha=0.7)
    plt.ylabel('Total Transmission Time (s)', fontsize=12)
    plt.title('TX Time by Network Tier', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s\n({height/energy["total_tx_time"]*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)


# ====================================
# MULTIACCESS & QUEUEING ANALYSIS
# ====================================
def analyze_system_performance(net, results):
    """Calculate multiaccess and queueing metrics"""
    
    energy = net.energy_stats
    trans = net.transmissions
    
    print("\n" + "="*70)
    print("MULTIACCESS ANALYSIS (Module 4)")
    print("="*70)
    
    # Channel utilization
    backbone_util = energy['backbone_tx_time'] / SIM_TIME if SIM_TIME > 0 else 0
    member_util = energy['member_tx_time'] / SIM_TIME if SIM_TIME > 0 else 0
    total_util = energy['total_tx_time'] / SIM_TIME if SIM_TIME > 0 else 0
    
    print(f"\nChannel Utilization:")
    print(f"  Backbone tier:      {backbone_util:.4f} ({backbone_util*100:.2f}%)")
    print(f"  Member tier:        {member_util:.4f} ({member_util*100:.2f}%)")
    print(f"  Total (both tiers): {total_util:.4f} ({total_util*100:.2f}%)")
    
    # Throughput analysis
    backbone_transmissions = sum(trans['is_backbone'])
    member_transmissions = len(trans['is_backbone']) - backbone_transmissions
    
    backbone_throughput = backbone_transmissions / SIM_TIME if SIM_TIME > 0 else 0
    member_throughput = member_transmissions / SIM_TIME if SIM_TIME > 0 else 0
    total_throughput = len(trans['distances']) / SIM_TIME if SIM_TIME > 0 else 0
    
    print(f"\nThroughput (transmissions/sec):")
    print(f"  Backbone tier:      {backbone_throughput:.2f} pkts/s")
    print(f"  Member tier:        {member_throughput:.2f} pkts/s")
    print(f"  Total:              {total_throughput:.2f} pkts/s")
    
    # TDMA slot efficiency
    num_heads = len(net.cluster_head)
    num_members = net.N - num_heads
    
    backbone_slots_available = SIM_TIME / BACKBONE_AIRTIME
    member_slots_available = SIM_TIME / MEMBER_AIRTIME if num_members > 0 else 1
    
    backbone_efficiency = backbone_transmissions / backbone_slots_available if backbone_slots_available > 0 else 0
    member_efficiency = member_transmissions / member_slots_available if member_slots_available > 0 else 0
    
    print(f"\nTDMA Slot Efficiency:")
    print(f"  Backbone slots available: {backbone_slots_available:.0f}")
    print(f"  Backbone slots used:      {backbone_transmissions}")
    print(f"  Backbone efficiency:      {backbone_efficiency:.4f} ({backbone_efficiency*100:.2f}%)")
    print(f"  Member slots available:   {member_slots_available:.0f}")
    print(f"  Member slots used:        {member_transmissions}")
    print(f"  Member efficiency:        {member_efficiency:.4f} ({member_efficiency*100:.2f}%)")
    
    # Success rate by tier
    backbone_success_rate = 0
    member_success_rate = 0
    if len(trans['is_backbone']) > 0:
        backbone_successes = sum([trans['successes'][i] for i, is_bb in enumerate(trans['is_backbone']) if is_bb])
        member_successes = sum([trans['successes'][i] for i, is_bb in enumerate(trans['is_backbone']) if not is_bb])
        
        backbone_success_rate = backbone_successes / backbone_transmissions if backbone_transmissions > 0 else 0
        member_success_rate = member_successes / member_transmissions if member_transmissions > 0 else 0
        
        print(f"\nSuccess Rates by Tier:")
        print(f"  Backbone tier:      {backbone_success_rate:.4f} ({backbone_success_rate*100:.2f}%)")
        print(f"  Member tier:        {member_success_rate:.4f} ({member_success_rate*100:.2f}%)")
    
    # Goodput (successful packets only)
    backbone_goodput = (backbone_transmissions * backbone_success_rate) / SIM_TIME if SIM_TIME > 0 and backbone_transmissions > 0 else 0
    member_goodput = (member_transmissions * member_success_rate) / SIM_TIME if SIM_TIME > 0 and member_transmissions > 0 else 0
    
    print(f"\nGoodput (successful pkts/sec):")
    print(f"  Backbone tier:      {backbone_goodput:.2f} pkts/s")
    print(f"  Member tier:        {member_goodput:.2f} pkts/s")
    
    print("\n" + "="*70)
    print("QUEUEING ANALYSIS (Module 3)")
    print("="*70)
    
    # Calculate utilization factors (ρ = λ/μ)
    avg_cluster_size = net.N / num_heads if num_heads > 0 else 1
    
    # Member nodes queueing
    lambda_member = 1.0  # 1 packet/sec per member (PRESSURE_PERIOD)
    mu_member = 1.0 / MEMBER_AIRTIME  # service rate (packets/sec)
    rho_member = lambda_member / mu_member
    
    print(f"\nMember Node Queueing:")
    print(f"  Arrival rate (λ):        {lambda_member:.3f} pkts/s per node")
    print(f"  Service rate (μ):        {mu_member:.3f} pkts/s")
    print(f"  Utilization factor (ρ):  {rho_member:.6f}")
    print(f"  System stable:           {'YES (ρ < 1)' if rho_member < 1 else 'NO (ρ >= 1) - UNSTABLE!'}")
    
    # Theoretical M/D/1 queue metrics (deterministic service time)
    L_member_theory = 0
    W_member_theory = 0
    if rho_member < 1:
        # Mean queue length: L = ρ + (ρ²)/(2(1-ρ))
        L_member_theory = rho_member + (rho_member**2) / (2 * (1 - rho_member))
        # Mean waiting time: W = L/λ (Little's Law)
        W_member_theory = L_member_theory / lambda_member
        print(f"  Theoretical mean queue:  {L_member_theory:.4f} packets")
        print(f"  Theoretical mean wait:   {W_member_theory:.4f} seconds")
    
    # Backbone nodes queueing (aggregated load)
    lambda_backbone = avg_cluster_size  # aggregate rate from cluster members
    mu_backbone = 1.0 / BACKBONE_AIRTIME
    rho_backbone = lambda_backbone / mu_backbone
    
    print(f"\nCluster Head (Backbone) Queueing:")
    print(f"  Avg cluster size:        {avg_cluster_size:.2f} nodes")
    print(f"  Arrival rate (λ):        {lambda_backbone:.3f} pkts/s (aggregated)")
    print(f"  Service rate (μ):        {mu_backbone:.3f} pkts/s")
    print(f"  Utilization factor (ρ):  {rho_backbone:.6f}")
    print(f"  System stable:           {'YES (ρ < 1)' if rho_backbone < 1 else 'NO (ρ >= 1) - UNSTABLE!'}")
    
    L_backbone_theory = 0
    W_backbone_theory = 0
    if rho_backbone < 1:
        L_backbone_theory = rho_backbone + (rho_backbone**2) / (2 * (1 - rho_backbone))
        W_backbone_theory = L_backbone_theory / lambda_backbone
        print(f"  Theoretical mean queue:  {L_backbone_theory:.4f} packets")
        print(f"  Theoretical mean wait:   {W_backbone_theory:.4f} seconds")
    
    # Empirical queue statistics
    print(f"\nEmpirical Queue Statistics:")
    
    member_nodes = [n for n in net.nodes if n.id not in net.cluster_head.values()]
    head_nodes = [n for n in net.nodes if n.id in net.cluster_head.values()]
    
    if member_nodes:
        all_member_samples = []
        for node in member_nodes:
            if len(node.queue_samples) > 0:
                all_member_samples.extend(node.queue_samples)
        
        if all_member_samples:
            mean_member_queue = np.mean(all_member_samples)
            max_member_queue = np.max(all_member_samples)
            print(f"  Member nodes:")
            print(f"    Mean queue length:     {mean_member_queue:.4f} packets")
            print(f"    Max queue length:      {max_member_queue:.0f} packets")
    
    if head_nodes:
        all_head_samples = []
        for node in head_nodes:
            if len(node.queue_samples) > 0:
                all_head_samples.extend(node.queue_samples)
        
        if all_head_samples:
            mean_head_queue = np.mean(all_head_samples)
            max_head_queue = np.max(all_head_samples)
            print(f"  Cluster heads:")
            print(f"    Mean queue length:     {mean_head_queue:.4f} packets")
            print(f"    Max queue length:      {max_head_queue:.0f} packets")
    
    # System-wide stability check
    print(f"\nSystem Stability Assessment:")
    if rho_member < 1 and rho_backbone < 1:
        print(f"  ✓ Both tiers stable (ρ < 1)")
        print(f"  ✓ System can sustain offered load")
    else:
        print(f"  ✗ UNSTABLE SYSTEM DETECTED")
        if rho_member >= 1:
            print(f"    - Member tier overloaded (ρ = {rho_member:.3f})")
        if rho_backbone >= 1:
            print(f"    - Backbone tier overloaded (ρ = {rho_backbone:.3f})")
    
    # Little's Law verification
    if results['delivered'] > 0 and len(results['latencies']) > 0:
        print(f"\nLittle's Law Verification:")
        mean_latency = np.mean(results['latencies'])
        delivery_rate = results['delivered'] / SIM_TIME
        predicted_queue = delivery_rate * mean_latency
        print(f"  Mean latency (W):        {mean_latency:.4f} seconds")
        print(f"  Delivery rate (λ):       {delivery_rate:.4f} pkts/s")
        print(f"  Predicted queue (L=λW):  {predicted_queue:.4f} packets")
    
    print("\n" + "="*70)
    print("ROUTING & PROTOCOL NOTES")
    print("="*70)
    print(f"\nRouting (Module 5):")
    print(f"  Cluster head selection:  Based on link reliability (probabilistic)")
    print(f"  Forwarding strategy:     Next-hop to adjacent cluster head")
    print(f"  Number of clusters:      {len(net.clusters)}")
    print(f"  Average hops end-to-end: {results['hops'].mean():.2f}" if len(results['hops']) > 0 else "  No data")
    
    print(f"\nProtocols (Module 2):")
    print(f"  Packet structure:        Aggregated sensor readings")
    print(f"  Aggregation period:      1.0 second per cluster")
    print(f"  Error recovery:          None (fresh data prioritized)")
    print(f"  Acknowledgements:        Implicit (no retransmission)")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics for plotting or further analysis
    return {
        'multiaccess': {
            'backbone_util': backbone_util,
            'member_util': member_util,
            'total_util': total_util,
            'backbone_throughput': backbone_throughput,
            'member_throughput': member_throughput,
            'backbone_efficiency': backbone_efficiency,
            'member_efficiency': member_efficiency,
            'backbone_success_rate': backbone_success_rate if backbone_transmissions > 0 else 0,
            'member_success_rate': member_success_rate if member_transmissions > 0 else 0,
        },
        'queueing': {
            'rho_member': rho_member,
            'rho_backbone': rho_backbone,
            'lambda_member': lambda_member,
            'mu_member': mu_member,
            'lambda_backbone': lambda_backbone,
            'mu_backbone': mu_backbone,
            'L_member_theory': L_member_theory if rho_member < 1 else float('inf'),
            'L_backbone_theory': L_backbone_theory if rho_backbone < 1 else float('inf'),
            'stable': (rho_member < 1 and rho_backbone < 1),
        }
    }


# ====================================
# RUN SIM
# ====================================
def run_sim(num_cars=NUM_CARS_DEFAULT, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    net = TrainNetwork(env, num_cars)
    env.run(until=SIM_TIME)

    lat = np.array(net.latencies)
    delivered = len(lat)
    # Total packets originated = (num_cars - 1) non-HOT nodes * SIM_TIME seconds
    originated = int((num_cars - 1) * SIM_TIME)

    results = {
        "num_cars": num_cars,
        "delivered": delivered,
        "originated": originated,
        "p_delivery": delivered / originated if originated > 0 else 0.0,
        "p_under_1s": float((lat <= 1.0).mean()) if delivered > 0 else 0.0,
        "latencies": lat,
        "hops": np.array(net.hops_record),
        "clusters": net.clusters,
        "heads": net.cluster_head,
    }
    
    print(f"\n=== Clustering Topology: {num_cars} cars ===")
    print(f"Clusters: {len(net.clusters)}")
    print(f"Cluster heads: {sorted(net.cluster_head.values())}")
    print(f"Packets originated: {originated}")
    print(f"Delivered: {delivered}")
    print(f"P(delivery): {results['p_delivery']:.3f}")
    if delivered > 0:
        print(f"P(lat<=1s | delivered): {results['p_under_1s']:.3f}")
        print(f"Mean latency: {lat.mean():.3f} s")
        print(f"Median latency: {np.median(lat):.3f} s")
        print(f"Mean hops: {results['hops'].mean():.2f}")

    return results, net


# ====================================
# MAIN
# ====================================
if __name__ == "__main__":
    results, net = run_sim(NUM_CARS_DEFAULT, seed=42)
    
    # Run the comprehensive analysis
    analysis_metrics = analyze_system_performance(net, results)

    if results["delivered"] > 0:
        plot_comprehensive_analysis(results, net)
        plt.show()  # This will display all figures at once
