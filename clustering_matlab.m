% Cluster TDMA Simulation Design
% Parameters
train_sizes = [10 20 50 100]; % # of Cars
CAR_LENGTH = 18; % Meters per Car
SLOT_TIME = 0.1; % sec
PACKETS_PER_NODE = 100; % # of Sensor Packets at Each Node
MAX_RANGE = 900; % Meters
DECAY_LAMBDA = 450; % Exponential Decay Term
REL_THRESH = 0.1; % Reliable Threshold Probability
INTERFERENCE_COEFF = 0.001; % More Nodes Increases the Interference
BASE_PKT_SIZE = 14; % Bytes per Packet
PKT_SIZE_ALPHA = 0.001; % Packet-Size Probability Factor
MAX_RETRIES = 3; % Try Resending a Dropped Pacekt 3 Times in QUEUE Mode
results = struct(); % store per train size
modes = {'drop','queue'};
for Ni = 1:length(train_sizes)
   N = train_sizes(Ni);
   fprintf('\n Train With: %d Cars\n', N);
   positions = (0:(N-1))' * CAR_LENGTH; % Position of Car With 0=Tail/EOT
   % Cluster Formation
   cluster_heads = [];
   cluster_members = {};
   i = 1;
   while i <= N
       furthest = i;
       for j = i+1:N
           d = positions(j) - positions(i);
           % Probability of Success Based on Distance
           p_dist = exp(-d/DECAY_LAMBDA) * (d < MAX_RANGE);
           % Packet Size Factor
           sf = 1 + PKT_SIZE_ALPHA * BASE_PKT_SIZE;
           % Interference Factor
           p_interf = exp(-INTERFERENCE_COEFF*N);
           % Combined Probability
           p = p_dist * exp(-d/(DECAY_LAMBDA/sf)) * p_interf;
           if p >= REL_THRESH
               furthest = j;
           else
               break;
           end
       end
       members = i:furthest;
       cluster_heads(end+1) = furthest;
       cluster_members{end+1} = members;
       i = furthest + 1;
   end
   num_clusters = length(cluster_heads);
   % Cluster Starts at the End Indices
   cluster_starts = [1, cluster_heads(1:end-1)+1];
   cluster_ends = cluster_heads;
   cluster_sizes = cluster_ends - cluster_starts + 1;
   cluster_pos = positions(cluster_heads);
  
   % Map Nodes to Cluster Index
   node2cluster = repelem(1:num_clusters, cluster_sizes)';
   % Main Simulation
   for m = 1:2
       mode = modes{m};
       dropped = 0;
       delivered = 0;
       total_tx_time = 0;
       total_latency = 0;
       hop_counts = [];
       for node = 1:N
           for pk = 1:PACKETS_PER_NODE
               % Node Goes to its Cluster Head
               cidx = node2cluster(node);
               head_idx = cluster_heads(cidx);
               % Probability From Disnance Interferene and Packet Size
               d1 = positions(head_idx) - positions(node);
               L1 = BASE_PKT_SIZE;
               sf = 1 + PKT_SIZE_ALPHA * L1;
               p_dist = exp(-d1 / DECAY_LAMBDA) * (d1 < MAX_RANGE);
               p_interf = exp(-INTERFERENCE_COEFF * N);
               p1 = p_dist * exp(-d1 / (DECAY_LAMBDA / sf)) * p_interf;
               success1 = false;
               if strcmp(mode,'drop')
                   % If a Random Number [0,1] is < the Computed
                   % Probability of Success then it is Successful
                   if rand < p1
                       success1 = true;
                       total_tx_time = total_tx_time + SLOT_TIME;
                       total_latency = total_latency + SLOT_TIME;
                   else % Else the Packet is just Dropped
                       dropped = dropped + 1;
                       total_tx_time = total_tx_time + SLOT_TIME;
                       continue;
                   end
               else % Else for The QUEUE Case
                   tries = 0;
                   while tries < MAX_RETRIES % Try to Resend Dropped Packets 3 Times
                       tries = tries + 1;
                       total_tx_time = total_tx_time + SLOT_TIME;
                       total_latency = total_latency + SLOT_TIME;
                       if rand < p1
                           success1 = true;
                           break;
                       end
                   end
                   if ~success1
                       dropped = dropped + 1; % Increment Dropped Number
                       continue;
                   end
               end
               % Cluster Forwarding Chain
               current_cluster = cidx;
               hop_count = 0;
               success_chain = true;
               while true
                   hop_count = hop_count + 1;
                   % Determines the Destination for the Hop
                   if current_cluster == num_clusters
                       dest_pos = positions(N);
                   else
                       dest_pos = cluster_pos(current_cluster+1);
                   end
                   src_pos = cluster_pos(current_cluster);
                   d2 = dest_pos - src_pos; % Hop Disntance
                   % Probability with Distance and Interference Terms
                   L2 = cluster_sizes(current_cluster) * BASE_PKT_SIZE;
                   sf = 1 + PKT_SIZE_ALPHA * L2;
                   p_dist = exp(-d2 / DECAY_LAMBDA) * (d2 < MAX_RANGE);
                   p_interf = exp(-INTERFERENCE_COEFF * N);
                   p2 = p_dist * exp(-d2 / (DECAY_LAMBDA / sf)) * p_interf;
                   if strcmp(mode,'drop')
                       total_tx_time = total_tx_time + SLOT_TIME;
                       if rand < p2 % Packet Successfully Forwarded Data
                           total_latency = total_latency + SLOT_TIME;
                           if current_cluster == num_clusters
                               break; % Reached HOT/Done
                           else % Go to Next Cluster
                               current_cluster = current_cluster + 1;
                           end
                       else % Packet Lost
                           dropped = dropped + 1;
                           success_chain = false;
                           break;
                       end
                   else % In QUEUE Mode and Retry Sending up to 3 Times per Hop
                       attempts = 0;
                       hop_ok = false;
                       while attempts < MAX_RETRIES
                           attempts = attempts + 1;
                           total_tx_time = total_tx_time + SLOT_TIME;
                           total_latency = total_latency + SLOT_TIME;
                           if rand < p2
                               hop_ok = true; % Successful Hop
                               break;
                           end
                       end
                       if ~hop_ok % Failed After 3 Retries
                           dropped = dropped + 1;
                           success_chain = false;
                           break;
                       else
                           if current_cluster == num_clusters
                               break; % Reached HOT/Done
                           else % Go to Next Cluser
                               current_cluster = current_cluster + 1;
                           end
                       end
                   end
               end
               % If the Chain was Successful, Save the Hop Data for
               % Plotting
               if success_chain
                   delivered = delivered + 1;
                   hop_counts(end+1) = hop_count;
               end
           end
       end
      
       % Save Results
       results(Ni).(mode).delivered = delivered;
       results(Ni).(mode).dropped = dropped;
       results(Ni).(mode).total_packets = N * PACKETS_PER_NODE;
       results(Ni).(mode).reliability = delivered / (N * PACKETS_PER_NODE);
       results(Ni).(mode).avg_latency = total_latency / max(delivered,1);
       results(Ni).(mode).total_tx_time = total_tx_time;
       results(Ni).(mode).avg_hops = mean(hop_counts);
       results(Ni).(mode).num_clusters = num_clusters;
      
       % Print Results
       fprintf('\n Mode: %s\n', mode);
       fprintf('Number of Packets: %d\n', results(Ni).(mode).total_packets);
       fprintf('Packets Delivered: %d, Packets Dropped: %d\n', results(Ni).(mode).delivered, results(Ni).(mode).dropped);
       fprintf('Reliability: %.3f\n', results(Ni).(mode).reliability);
       fprintf('Avg Latency (s): %.3f\n', results(Ni).(mode).avg_latency);
       fprintf('Avg Hops: %.2f\n', results(Ni).(mode).avg_hops);
       fprintf('Clusters: %d\n', results(Ni).(mode).num_clusters);
   end
end
% Pre Size the Arrays for the Results for Plotting
distances = train_sizes * CAR_LENGTH;
rel_drop = zeros(1,length(train_sizes));
rel_queue = zeros(1,length(train_sizes));
lat_drop = zeros(1,length(train_sizes));
lat_queue = zeros(1,length(train_sizes));
hops_drop = zeros(1,length(train_sizes));
hops_queue = zeros(1,length(train_sizes));
cluster_counts = zeros(1,length(train_sizes));
energy_drop = zeros(1,length(train_sizes));
energy_queue = zeros(1,length(train_sizes));
% Extract the Results from the Results struct
for i = 1:length(train_sizes)
   rel_drop(i) = results(i).drop.reliability;
   rel_queue(i) = results(i).queue.reliability;
   lat_drop(i) = results(i).drop.avg_latency;
   lat_queue(i) = results(i).queue.avg_latency;
   hops_drop(i) = results(i).drop.avg_hops;
   hops_queue(i) = results(i).queue.avg_hops;
   cluster_counts(i) = results(i).drop.num_clusters;
   energy_drop(i) = results(i).drop.total_tx_time;
   energy_queue(i) = results(i).queue.total_tx_time;
end
% Channel probability degradation = 1 - reliability
prob_deg_drop = 1 - rel_drop;
prob_deg_queue = 1 - rel_queue;
% Plots
figure;
plot(distances, rel_drop,'-o','LineWidth',1.8);
hold on;
plot(distances, rel_queue,'-s','LineWidth',1.8);
xlabel('Train Length (m)');
ylabel('EOT to HOT Reliability');
legend('DROP','QUEUE');
grid on;
title('Reliability vs Train Length');
figure;
plot(distances, lat_drop,'-o','LineWidth',1.8);
hold on;
plot(distances, lat_queue,'-s','LineWidth',1.8);
xlabel('Train Length (m)');
ylabel('Avg Latency (s)');
legend('DROP','QUEUE');
grid on;
title('Latency vs Train Length');
figure;
plot(distances, hops_drop,'-o','LineWidth',1.8);
hold on;
plot(distances, hops_queue,'-s','LineWidth',1.8);
xlabel('Train Length (m)');
ylabel('Avg Hops');
legend('DROP','QUEUE');
grid on; title('Average Hops');
figure;
bar(train_sizes, cluster_counts);
xlabel('Number of cars');
ylabel('Number of clusters');
title('Clusters formed');
grid on;
figure;
plot(distances, energy_drop,'-o','LineWidth',1.8);
hold on;
plot(distances, energy_queue,'-s','LineWidth',1.8);
xlabel('Train Length (m)');
ylabel('Energy (arbitrary units)');
legend('DROP','QUEUE');
grid on;
title('Energy Consumption vs Train Length');
% Channel Degredation Plot and Probability of Success
% Using Equations Used in the Sim
d = 0:10:1200;            
L = BASE_PKT_SIZE;         
N = max(train_sizes);      
sf = 1 + PKT_SIZE_ALPHA * L;
p_interf = exp(-INTERFERENCE_COEFF * N);
P_dist = exp(-d / DECAY_LAMBDA) .* (d < MAX_RANGE);
P_success = P_dist .* exp(-d / (DECAY_LAMBDA / sf)) * p_interf;
channel_deg = 1 - P_success;
% Plot
figure;
plot(d, P_success,'LineWidth',1.8);
hold on;
plot(d, channel_deg,'LineWidth',1.8);
xlabel('Distance (m)');
ylabel('Probability');
title('Per-Hop Success vs Channel Degradation');
legend('P of Success','Channel Degradation');
grid on;
ylim([0 1]);
