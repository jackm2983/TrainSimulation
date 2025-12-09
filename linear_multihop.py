import simpy
import numpy as np
import random
import matplotlib.pyplot as plt

# =========================
# Model parameters
# =========================
NUM_CARS_DEFAULT = 100          # Number of nodes on the train
SLOT_DURATION = 0.01            # TDMA slot length [s]; 50ms realistic for LoRa SF7
PRESSURE_PERIOD = 1.0           # Each node generates one new reading every 1 s
SIM_TIME = 600.0                # Total simulation time [s]
MAX_LORA_RANGE = 900.0          # Max reliable LoRa distance [m]
CAR_LENGTH_MIN = 20.0           # Min freight car length [m]
CAR_LENGTH_MAX = 40.0           # Max freight car length [m]
RELIABILITY_THRESHOLD = 0.95    # Min per-link success probability to be considered "reliable"
CHANNEL_DEGRADATION = 1.0       # Scale factor (0-1) to degrade all links
INTERFERENCE_DROP_PROB = 0.0    # Extra drop probability from other trains, etc.
VERBOSE = False                 # Print per-slot events (can be very noisy)


class Packet:
    """Represents a single pressure reading moving toward HOT."""
    _next_id = 0

    def __init__(self, origin_id, created_time):
        self.id = Packet._next_id
        Packet._next_id += 1
        self.origin_id = origin_id
        self.created_time = created_time
        self.hops = 0


class Node:
    """One car: pressure sensor + relay."""
    def __init__(self, env, node_id, position, network, is_eot=False, is_hot=False):
        self.env = env
        self.id = node_id
        self.position = position
        self.network = network
        self.is_eot = is_eot
        self.is_hot = is_hot

        # Queue for packets (own + relayed mixed, FIFO with fresh data priority via dropping old)
        self.packet_queue = []

        # Stats
        self.queue_length_samples = []
        self.queue_sample_times = []
        self.packets_sent = 0
        self.packets_dropped = 0
        self.packets_generated = 0
        self.packets_relayed = 0

        if not self.is_hot:
            # Every node (including EOT) generates sensor readings at 1 Hz
            self.env.process(self.generate_readings())

    # ---------- Traffic generation ----------
    def generate_readings(self):
        """Generate local pressure readings periodically and enqueue them."""
        while True:
            yield self.env.timeout(PRESSURE_PERIOD)
            pkt = Packet(origin_id=self.id, created_time=self.env.now)
            self.packet_queue.append(pkt)
            self.packets_generated += 1

    def enqueue_relay(self, pkt):
        """Receive a relayed packet from upstream."""
        self.packet_queue.append(pkt)
        self.packets_relayed += 1

    def sample_queue_length(self):
        """Record instantaneous queue length."""
        self.queue_length_samples.append(len(self.packet_queue))
        self.queue_sample_times.append(self.env.now)

    # ---------- TDMA transmission ----------
    def send_in_slot(self, slot_index):
        """Called by TDMA scheduler when this node is allowed to transmit."""
        # Record queue length sample
        self.sample_queue_length()

        if not self.packet_queue:
            return

        pkt = self.packet_queue.pop(0)

        # HOT node is a sink: if it somehow has a queued packet, treat that as arrival
        if self.is_hot:
            self.network.record_delivery(pkt, self)
            return

        # Choose next hop toward HOT (furthest reliable receiver)
        next_hop_id, distance, p_success = self.network.choose_next_hop(self.id)
        if next_hop_id is None:
            # Nowhere to send; drop packet
            self.packets_dropped += 1
            if VERBOSE:
                print(f"[{self.env.now:8.3f}s] Slot {slot_index:6d}: "
                      f"Node {self.id:3d} has no forward link, dropping pkt {pkt.id}")
            return

        # Apply channel reliability, degradation, and random interference
        p_link = p_success * CHANNEL_DEGRADATION * (1.0 - INTERFERENCE_DROP_PROB)
        p_link = max(0.0, min(1.0, p_link))

        success = (random.random() < p_link)

        # Record transmission attempt
        self.network.record_transmission(self.id, next_hop_id, distance, p_link, success)

        if VERBOSE:
            print(f"[{self.env.now:8.3f}s] Slot {slot_index:6d}: "
                  f"Node {self.id:3d} -> {next_hop_id:3d} "
                  f"dist={distance:6.1f}m "
                  f"Psucc={p_link:4.2f} "
                  f"{'OK' if success else 'DROP'} for pkt {pkt.id}")

        if success:
            pkt.hops += 1
            self.packets_sent += 1
            receiver = self.network.nodes[next_hop_id]
            # If this hop delivers to HOT, record stats
            if receiver.is_hot:
                self.network.record_delivery(pkt, receiver)
            else:
                receiver.enqueue_relay(pkt)
        else:
            # Packet lost forever (packet-erasure model, no retransmission)
            self.packets_dropped += 1


class TrainNetwork:
    """Full LoRa multi-hop TDMA network on a single train."""
    def __init__(self, env, num_cars):
        self.env = env
        self.num_cars = num_cars
        self.slot_duration = SLOT_DURATION

        # Layout
        self.positions = self._generate_positions(num_cars)
        self.eot_id = 0
        self.hot_id = num_cars - 1

        # Nodes
        self.nodes = []
        for i in range(num_cars):
            is_eot = (i == self.eot_id)
            is_hot = (i == self.hot_id)
            node = Node(env, i, self.positions[i], self, is_eot=is_eot, is_hot=is_hot)
            self.nodes.append(node)

        # Stats for all packets
        self.latencies = []          # list of delivery latencies
        self.hops_per_packet = []    # hops per delivered packet
        self.latency_by_origin = {i: [] for i in range(num_cars)}
        
        # Transmission statistics
        self.transmissions = {
            'distances': [],
            'probabilities': [],
            'successes': [],
            'src_nodes': [],
            'dst_nodes': []
        }

        # Start TDMA scheduler
        self.env.process(self.tdma_scheduler())

    # ---------- Layout and channel ----------
    def _generate_positions(self, num_cars):
        """Randomly generate car positions based on length distribution."""
        car_lengths = np.random.uniform(CAR_LENGTH_MIN, CAR_LENGTH_MAX, size=num_cars)
        positions = np.zeros(num_cars)
        for i in range(1, num_cars):
            positions[i] = positions[i - 1] + car_lengths[i - 1]
        return positions

    def link_success_prob(self, distance):
        """Packet success probability vs distance (simple exponential decay)."""
        if distance <= 0 or distance > MAX_LORA_RANGE:
            return 0.0
        # Choose a decay so that short hops are ~1.0 and at MAX_LORA_RANGE ~0.3
        decay_distance = 1500.0
        p = np.exp(-(distance / decay_distance) ** 2)
        return float(max(0.0, min(1.0, p)))

    def choose_next_hop(self, src_id):
        """Select furthest reliable receiver toward HOT, with fallback to closest reachable."""
        src_pos = self.positions[src_id]
        best_id = None
        best_dist = 0.0
        best_p = 0.0

        # First pass: furthest node with reliability above threshold
        for j in range(src_id + 1, self.num_cars):
            d = self.positions[j] - src_pos
            if d <= 0 or d > MAX_LORA_RANGE:
                continue
            p = self.link_success_prob(d)
            if p >= RELIABILITY_THRESHOLD and d > best_dist:
                best_id = j
                best_dist = d
                best_p = p

        if best_id is not None:
            return best_id, best_dist, best_p

        # Fallback: nearest in-front neighbor within range, even if below threshold
        for j in range(src_id + 1, self.num_cars):
            d = self.positions[j] - src_pos
            if d <= 0 or d > MAX_LORA_RANGE:
                continue
            p = self.link_success_prob(d)
            return j, d, p

        # No forward link
        return None, None, 0.0

    # ---------- Stats ----------
    def record_transmission(self, src, dst, distance, prob, success):
        """Record transmission attempt statistics."""
        self.transmissions['distances'].append(distance)
        self.transmissions['probabilities'].append(prob)
        self.transmissions['successes'].append(1 if success else 0)
        self.transmissions['src_nodes'].append(src)
        self.transmissions['dst_nodes'].append(dst)

    def record_delivery(self, pkt, sink_node):
        """Called whenever a packet reaches HOT."""
        latency = self.env.now - pkt.created_time
        self.latencies.append(latency)
        self.hops_per_packet.append(pkt.hops)
        self.latency_by_origin[pkt.origin_id].append(latency)

    # ---------- TDMA scheduler ----------
    def tdma_scheduler(self):
        """Global TDMA schedule rotating through all cars."""
        slot_index = 0
        while self.env.now < SIM_TIME:
            node_id = slot_index % self.num_cars
            node = self.nodes[node_id]
            node.send_in_slot(slot_index)
            slot_index += 1
            yield self.env.timeout(self.slot_duration)


# ====================================
# VISUALIZATION
# ====================================
def plot_comprehensive_analysis(results, net):
    """Create comprehensive visualization of system behavior"""
    
    # Calculate common data
    distances = np.linspace(0, MAX_LORA_RANGE, 1000)
    probabilities = [net.link_success_prob(d) for d in distances]
    trans = net.transmissions
    
    # 1. Signal Degradation Model
    plt.figure(figsize=(10, 6))
    plt.plot(distances, probabilities, 'b-', linewidth=2)
    plt.axhline(y=RELIABILITY_THRESHOLD, color='r', linestyle='--', 
                label=f'Threshold ({RELIABILITY_THRESHOLD})')
    plt.axvline(x=MAX_LORA_RANGE, color='g', linestyle='--', alpha=0.5, 
                label=f'Max Range ({MAX_LORA_RANGE}m)')
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title('LoRa Link Success vs Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 2. Train Layout
    plt.figure(figsize=(14, 6))
    plt.scatter(net.positions, np.zeros(net.num_cars), s=50, alpha=0.6, c='blue')
    plt.scatter(net.positions[net.eot_id], 0, s=300, marker='s', 
               c='red', edgecolors='black', linewidths=2, label='EOT', zorder=5)
    plt.scatter(net.positions[net.hot_id], 0, s=300, marker='*', 
               c='green', edgecolors='black', linewidths=2, label='HOT', zorder=5)
    plt.xlabel('Position along train (m)', fontsize=12)
    plt.title(f'Train Layout: {net.num_cars} cars, Length: {net.positions[-1]:.1f}m', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.yticks([])
    
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
        plt.scatter(trans['distances'], trans['successes'], alpha=0.3, s=10, c='blue')
        plt.xlabel('Distance (m)', fontsize=12)
        plt.ylabel('Success (1) / Failure (0)', fontsize=12)
        plt.title('Transmission Outcomes by Distance', fontsize=14, fontweight='bold')
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
    
    # 8. Queue Length Over Time (sample nodes)
    plt.figure(figsize=(12, 6))
    sample_nodes = [0, net.num_cars//4, net.num_cars//2, 3*net.num_cars//4, net.num_cars-2]
    for node_id in sample_nodes:
        if node_id < net.num_cars and len(net.nodes[node_id].queue_length_samples) > 0:
            plt.plot(net.nodes[node_id].queue_sample_times, 
                    net.nodes[node_id].queue_length_samples,
                    alpha=0.7, linewidth=1.5, label=f'Node {node_id}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Queue Length', fontsize=12)
    plt.title('Queue Evolution (Sample Nodes)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 9. Per-Node Statistics
    plt.figure(figsize=(14, 6))
    node_ids = range(net.num_cars)
    packets_generated = [net.nodes[i].packets_generated for i in node_ids if not net.nodes[i].is_hot]
    packets_sent = [net.nodes[i].packets_sent for i in node_ids]
    packets_dropped = [net.nodes[i].packets_dropped for i in node_ids]
    
    x = np.arange(len(node_ids))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(node_ids, packets_sent, alpha=0.7, color='green', label='Sent')
    plt.bar(node_ids, packets_dropped, bottom=packets_sent, alpha=0.7, color='red', label='Dropped')
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Packet Count', fontsize=12)
    plt.title('Per-Node Transmission Statistics', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    delivery_counts = [len(net.latency_by_origin[i]) for i in node_ids]
    expected_per_node = SIM_TIME if not net.nodes[i].is_hot else 0
    delivery_rates = [c / expected_per_node if expected_per_node > 0 else 0 for c in delivery_counts]
    plt.bar(node_ids, delivery_rates, alpha=0.7, color='teal')
    plt.axhline(y=results['p_delivery'], color='r', linestyle='--', linewidth=2,
                label=f'Overall: {results["p_delivery"]:.3f}')
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Delivery Rate', fontsize=12)
    plt.title('Per-Node Delivery Success', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')


# ====================================
# MULTIACCESS & QUEUEING ANALYSIS
# ====================================
def analyze_system_performance(net, results):
    """Calculate multiaccess and queueing metrics for linear multi-hop TDMA"""
    
    trans = net.transmissions
    
    print("\n" + "="*70)
    print("MULTIACCESS ANALYSIS (Module 4)")
    print("="*70)
    
    # Channel utilization
    total_slots_available = SIM_TIME / SLOT_DURATION
    total_transmissions = len(trans['distances'])
    channel_util = total_transmissions / total_slots_available if total_slots_available > 0 else 0
    
    print(f"\nChannel Utilization:")
    print(f"  Total slots available:  {total_slots_available:.0f}")
    print(f"  Slots with transmission: {total_transmissions}")
    print(f"  Channel utilization:    {channel_util:.4f} ({channel_util*100:.2f}%)")
    
    # Throughput analysis
    throughput = total_transmissions / SIM_TIME if SIM_TIME > 0 else 0
    successful_transmissions = sum(trans['successes'])
    goodput = successful_transmissions / SIM_TIME if SIM_TIME > 0 else 0
    
    print(f"\nThroughput:")
    print(f"  Total transmissions/sec:     {throughput:.2f} pkts/s")
    print(f"  Successful transmissions/sec: {goodput:.2f} pkts/s")
    print(f"  Overall success rate:        {np.mean(trans['successes']):.4f} ({np.mean(trans['successes'])*100:.2f}%)")
    
    # TDMA slot efficiency (what fraction of slots actually transmit)
    slots_used_per_node = []
    for node in net.nodes:
        slots_used_per_node.append(node.packets_sent + node.packets_dropped)
    
    print(f"\nTDMA Slot Efficiency:")
    print(f"  Mean slots used per node:    {np.mean(slots_used_per_node):.1f}")
    print(f"  Nodes transmitting actively: {sum(1 for x in slots_used_per_node if x > 0)}/{net.num_cars}")
    
    # Per-node load analysis
    total_generated = sum(node.packets_generated for node in net.nodes if not node.is_hot)
    total_relayed = sum(node.packets_relayed for node in net.nodes)
    total_sent = sum(node.packets_sent for node in net.nodes)
    total_dropped = sum(node.packets_dropped for node in net.nodes)
    
    print(f"\nPacket Flow:")
    print(f"  Total packets generated:     {total_generated}")
    print(f"  Total relay receptions:      {total_relayed}")
    print(f"  Total transmissions (sent):  {total_sent}")
    print(f"  Total dropped (failures):    {total_dropped}")
    print(f"  Loss rate:                   {total_dropped/(total_sent+total_dropped):.4f} ({total_dropped/(total_sent+total_dropped)*100:.2f}%)")
    
    print("\n" + "="*70)
    print("QUEUEING ANALYSIS (Module 3)")
    print("="*70)
    
    # Calculate utilization factors (ρ = λ/μ)
    # For each node: arrival rate = generation rate + relay rate
    # Service rate = 1 / SLOT_DURATION (but shared among all nodes in TDMA)
    
    lambda_generation = 1.0  # 1 packet/sec per node (PRESSURE_PERIOD)
    mu_node = 1.0 / SLOT_DURATION  # service rate if node had dedicated channel
    mu_tdma = mu_node / net.num_cars  # effective service rate under TDMA sharing
    
    print(f"\nSingle Node Queueing (if isolated):")
    print(f"  Arrival rate (λ):        {lambda_generation:.3f} pkts/s per node")
    print(f"  Service rate (μ):        {mu_node:.3f} pkts/s")
    print(f"  Utilization factor (ρ):  {lambda_generation/mu_node:.6f}")
    print(f"  System stable:           {'YES (ρ < 1)' if lambda_generation/mu_node < 1 else 'NO'}")
    
    print(f"\nTDMA Shared Channel:")
    print(f"  Number of nodes sharing: {net.num_cars}")
    print(f"  Effective service rate:  {mu_tdma:.3f} pkts/s per node")
    print(f"  Utilization factor (ρ):  {lambda_generation/mu_tdma:.6f}")
    print(f"  System stable:           {'YES (ρ < 1)' if lambda_generation/mu_tdma < 1 else 'NO (ρ >= 1) - UNSTABLE!'}")
    
    # Theoretical M/D/1 queue metrics for TDMA
    rho_tdma = lambda_generation / mu_tdma
    if rho_tdma < 1:
        # Mean queue length: L = ρ + (ρ²)/(2(1-ρ))
        L_theory = rho_tdma + (rho_tdma**2) / (2 * (1 - rho_tdma))
        # Mean waiting time: W = L/λ (Little's Law)
        W_theory = L_theory / lambda_generation
        print(f"  Theoretical mean queue:  {L_theory:.4f} packets")
        print(f"  Theoretical mean wait:   {W_theory:.4f} seconds")
    else:
        print(f"  WARNING: System unstable (ρ >= 1), queues will grow unbounded!")
        L_theory = float('inf')
        W_theory = float('inf')
    
    # Empirical queue statistics
    print(f"\nEmpirical Queue Statistics:")
    all_queue_samples = []
    for node in net.nodes:
        if len(node.queue_length_samples) > 0:
            all_queue_samples.extend(node.queue_length_samples)
    
    if all_queue_samples:
        mean_queue = np.mean(all_queue_samples)
        max_queue = np.max(all_queue_samples)
        print(f"  Mean queue length (all nodes): {mean_queue:.4f} packets")
        print(f"  Max queue length (any node):   {max_queue:.0f} packets")
        
        # EOT specific (node 0)
        if len(net.nodes[net.eot_id].queue_length_samples) > 0:
            eot_mean = np.mean(net.nodes[net.eot_id].queue_length_samples)
            eot_max = np.max(net.nodes[net.eot_id].queue_length_samples)
            print(f"  EOT mean queue length:         {eot_mean:.4f} packets")
            print(f"  EOT max queue length:          {eot_max:.0f} packets")
    
    # System-wide stability check
    print(f"\nSystem Stability Assessment:")
    if rho_tdma < 1:
        print(f"  ✓ System stable (ρ = {rho_tdma:.3f} < 1)")
        print(f"  ✓ TDMA schedule can sustain offered load")
    else:
        print(f"  ✗ UNSTABLE SYSTEM (ρ = {rho_tdma:.3f} >= 1)")
        print(f"    - TDMA slots are insufficient for offered load")
        print(f"    - Queues will grow unbounded")
        print(f"    - Need faster TDMA cycle or more bandwidth")
    
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
    print(f"  Strategy:                Furthest reliable neighbor (greedy)")
    print(f"  Reliability threshold:   {RELIABILITY_THRESHOLD}")
    print(f"  Average hops end-to-end: {results['hops'].mean():.2f}" if len(results['hops']) > 0 else "  No data")
    print(f"  Max theoretical range:   {MAX_LORA_RANGE:.0f} m")
    
    print(f"\nProtocols (Module 2):")
    print(f"  MAC:                     Global TDMA ({SLOT_DURATION*1000:.0f}ms slots)")
    print(f"  Packet structure:        14 bytes (timestamp + GPS + pressure)")
    print(f"  Error recovery:          None (fresh data prioritized)")
    print(f"  Acknowledgements:        None (packet-erasure model)")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics
    return {
        'multiaccess': {
            'channel_util': channel_util,
            'throughput': throughput,
            'goodput': goodput,
            'success_rate': np.mean(trans['successes']),
        },
        'queueing': {
            'rho_tdma': rho_tdma,
            'lambda': lambda_generation,
            'mu_tdma': mu_tdma,
            'L_theory': L_theory,
            'W_theory': W_theory,
            'stable': (rho_tdma < 1),
        }
    }


# ====================================
# RUN SIM
# ====================================
def run_sim(num_cars=NUM_CARS_DEFAULT, seed=1234):
    random.seed(seed)
    np.random.seed(seed)

    env = simpy.Environment()
    network = TrainNetwork(env, num_cars)
    env.run(until=SIM_TIME)

    latencies = np.array(network.latencies)
    delivered = len(latencies)

    # Total packets originated = (num_cars - 1) non-HOT nodes * SIM_TIME seconds
    originated = int((num_cars - 1) * SIM_TIME)
    p_delivery = delivered / originated if originated > 0 else 0.0
    p_under_1s = float((latencies <= 1.0).mean()) if delivered > 0 else 0.0

    results = {
        "num_cars": num_cars,
        "delivered": delivered,
        "originated": originated,
        "p_delivery": p_delivery,
        "p_under_1s": p_under_1s,
        "latencies": latencies,
        "hops": np.array(network.hops_per_packet),
    }
    
    print(f"\n=== Linear Multi-Hop TDMA: {num_cars} cars ===")
    print(f"Train length: {network.positions[-1]:.1f} m")
    print(f"TDMA slot duration: {SLOT_DURATION*1000:.0f} ms")
    print(f"TDMA frame duration: {SLOT_DURATION*num_cars:.3f} s")
    print(f"Packets originated: {originated}")
    print(f"Delivered to HOT: {delivered}")
    print(f"P(delivery): {p_delivery:.3f}")
    if delivered > 0:
        print(f"P(lat<=1s | delivered): {p_under_1s:.3f}")
        print(f"Mean latency: {latencies.mean():.3f} s")
        print(f"Median latency: {np.median(latencies):.3f} s")
        print(f"Mean hops: {results['hops'].mean():.2f}")

    return results, network


# ====================================
# MAIN
# ====================================
if __name__ == "__main__":
    results, net = run_sim(NUM_CARS_DEFAULT, seed=42)
    
    # Run the comprehensive analysis
    analysis_metrics = analyze_system_performance(net, results)

    if results["delivered"] > 0:
        plot_comprehensive_analysis(results, net)
        plt.show()
