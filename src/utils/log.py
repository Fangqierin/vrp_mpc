import csv 

def log_route_plan(time, routes):
    # Column headers
    route_file = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/results/route_planning.csv"
    # Open the CSV file in write mode ('w'), with newline='' to prevent extra blank lines
    rows_to_write = []
    for vehicle_id, v_route in routes.items():
        for data in v_route:
            # Create a row with the vehicle's data
            vehicle_info = [
                time,
                vehicle_id,  # vehicle_id
                data[0],
                data[1],  # node_id
                data[2],  # arrival_time
                data[3],  # load
                data[4],  # travel_time
                data[5],  # early_arrival
                data[6],  # node
                data[7],  # type
            ]
            rows_to_write.append(vehicle_info)
    with open(route_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_write)  # Write all rows at once
