from collections import defaultdict


def update_state(
    new_time,
    a,
    points,
    n,
    depot_start_location,
    all_routes,
    idle_v,
):
    # new_time=1790
    P_N = n
    unvisited_nodes = defaultdict(list)
    NODE_ID_INDEX = 0
    ARRIVAL_TIME_INDEX = 1
    load_index = 2
    TRAVL_TIME_INDEX = 3
    A_INDEX = 4
    NODE_SEQ_INDEX = 5
    TYPE_INDEX = 6
    CHAIN_ORDER_INDEX = 7
    finished_request = []
    v_start_time = {}

    for k, route_v in all_routes.items():
        visited_nodes = []
        # Iterate through nodes and their times
        OD_dropoff_request = []
        for plnode in route_v:
            node_time = plnode[ARRIVAL_TIME_INDEX]
            if node_time <= new_time:
                visited_nodes.append(plnode)
            else:
                unvisited_nodes[k].append(plnode)
        # Sort the nodes (if needed, based on order in the route)
        visited_nodes = sorted(visited_nodes, key=lambda x: x[ARRIVAL_TIME_INDEX])  # Sorting by time
        unvisited_nodes[k] = sorted(unvisited_nodes[k], key=lambda x: x[ARRIVAL_TIME_INDEX])
        v_start_time[k] = new_time  # Update current time
        if len(visited_nodes) > 1:  # If visited node greater than 0, then update depot start location.
            v_current_node = visited_nodes[-1]
            if len(unvisited_nodes[k]) > 1:
                next_node = unvisited_nodes[k][0]
                if new_time + v_current_node[TRAVL_TIME_INDEX] >= next_node[A_INDEX] or next_node[TYPE_INDEX] == "RS":
                    v_current_node = next_node
                    v_new_time = next_node[ARRIVAL_TIME_INDEX]
                    v_start_time[k] = v_new_time
                    # unvisited_nodes[k] = unvisited_nodes[k]
            depot_start_location[k] = v_current_node[NODE_ID_INDEX]
        all_unvisited = [(i[NODE_SEQ_INDEX], i[CHAIN_ORDER_INDEX], i[TYPE_INDEX]) for i in unvisited_nodes[k]]
        unvisited_node_id = []
        for info in visited_nodes:
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = info
            if type == "D":
                finished_request.append(chain_order)
        for info in unvisited_nodes[k]:
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = info
            if type == "P":
                unvisited_node_id.append(node)
                unvisited_node_id.append(node + P_N)
        for info in all_unvisited:
            node_id, chain_order, type = info
            if node_id not in unvisited_node_id and type == "D":
                OD_dropoff_request.append([points[node_id][0], a[node_id] + 15 * 60, -1, k, points[node_id][1]])
    return (v_start_time, OD_dropoff_request, depot_start_location, finished_request)
