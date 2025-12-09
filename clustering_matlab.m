clear; clc; close all;

%% ---------------- Simulation Parameters -----------------------
train_sizes = [10 20 50 100];   % cars
cluster_radius = 30;            % meters
car_spacing = 10;               % meters
packets_per_node = 600;         % per run
tx_time = 0.1;                  % seconds per transmission
slot_time = 0.1;                % seconds
HOT_position = 0;               % head of train

% Loss model parameters
base_loss = 0.02;               % constant loss floor
loss_per_meter = 0.0005;        % proportional to distance
interference_coeff = 0.001;     % per node density

%% Arrays for plots
all_reliability = [];
all_latency = [];
all_energy = [];
all_distances = train_sizes * car_spacing;

%% ----------------------- Main Loop ----------------------------
for N = train_sizes

    fprintf("\n================ SIM SUMMARY ================\n");
    fprintf("Train size: %d cars\n", N);

    %% Generate node positions
    x = (1:N)' * car_spacing;

    %% ---- Cluster Formation ----
    clusters = {};
    cluster_heads = [];

    visited = false(N,1);

    for i = 1:N
        if visited(i), continue; end
        % Start a new cluster at node i
        cluster = i;
        visited(i) = true;

        % Add all nodes within cluster_radius
        for j = i+1:N
            if ~visited(j) && abs(x(j) - x(i)) <= cluster_radius
                cluster(end+1) = j;
                visited(j) = true;
            end
        end
        clusters{end+1} = cluster;
        % cluster head = physically closest to HOT (min x)
        [~, idxMin] = min(x(cluster));
        cluster_heads(end+1) = cluster(idxMin);
    end

    num_clusters = length(clusters);
    fprintf("Number of clusters: %d\n", num_clusters);

    %% ---------------- Simulate Packets ----------------
    total_packets = N * packets_per_node;
    node_to_cluster_delivered = 0;
    cluster_to_hot_delivered = 0;

    hops_cluster = [];
    hops_hot = [];
    latency_cluster = [];
    latency_hot = [];
    energy_used = [];

    for node = 1:N
        for p = 1:packets_per_node

            %% ---- Node → Cluster Head ----
            cluster_id = find(cellfun(@(c) any(c==node), clusters));
            head = cluster_heads(cluster_id);

            hop_count = 1; % simplified model (1 hop)
            lat = slot_time;

            % Loss probability increases with distance
            d = abs(x(node) - x(head));
            P_loss = base_loss + d*loss_per_meter + interference_coeff*N;

            if rand() > P_loss
                node_to_cluster_delivered = node_to_cluster_delivered + 1;
                hops_cluster(end+1) = hop_count;
                latency_cluster(end+1) = lat;
                energy_used(end+1) = tx_time;
            else
                continue; % packet lost, skip inter-cluster
            end

            %% ---- Cluster Head → HOT ----
            d2 = abs(x(head) - HOT_position);
            hop_count2 = 1 + (cluster_id - 1);    % simplified model
            lat2 = hop_count2 * slot_time;

            % Loss model again
            P_loss2 = base_loss + d2*loss_per_meter + interference_coeff*N;

            if rand() > P_loss2
                cluster_to_hot_delivered = cluster_to_hot_delivered + 1;
                hops_hot(end+1) = hop_count2;
                latency_hot(end+1) = lat2;
                energy_used(end+1) = tx_time * hop_count2;
            end

        end
    end

    %% ---------------- Summary Output ----------------
    fprintf("Packets generated: %d\n", total_packets);
    fprintf("Node → Cluster delivered: %d\n", node_to_cluster_delivered);
    fprintf("Cluster → HOT delivered: %d\n", cluster_to_hot_delivered);
    fprintf("Avg Node→Cluster hops: %.2f\n", mean(hops_cluster));
    fprintf("Avg Cluster→HOT hops: %.2f\n", mean(hops_hot));
    fprintf("Avg Node→Cluster latency: %.3f s\n", mean(latency_cluster));
    fprintf("Avg Cluster→HOT latency: %.3f s\n", mean(latency_hot));
    fprintf("--------------------------------------------\n");

    %% ---- Save for plots ----
    all_reliability(end+1) = cluster_to_hot_delivered / total_packets;
    all_latency(end+1) = mean(latency_hot);
    all_energy(end+1) = sum(energy_used);

end


%% ===================== PLOTS ======================

figure; 
plot(all_distances, all_reliability, '-o', 'LineWidth', 2);
xlabel('Train Length (m)');
ylabel('Reliability');
title('Reliability vs Distance');
grid on;

figure;
plot(all_distances, all_latency, '-o', 'LineWidth', 2);
xlabel('Train Length (m)');
ylabel('Avg Latency to HOT (s)');
title('Latency vs Distance');
grid on;

figure;
plot(all_distances, all_energy, '-o', 'LineWidth', 2);
xlabel('Train Length (m)');
ylabel('Total Energy Used (arbitrary units)');
title('Energy Use vs Distance');
grid on;

