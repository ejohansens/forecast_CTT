import psycopg2
import json
import pytz
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import json

# Database connection details (fill in yours)
DB_NAME = "your_new_database" 
DB_USER = "postgres"
DB_PASSWORD = "password123"
DB_HOST = "localhost"
DB_PORT = "5432"

# Define Dutch timezone
DUTCH_TZ = pytz.timezone("Europe/Amsterdam")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Converting to dutch time
def convert_to_dutch_time(dt):
    """Convert UTC datetime to Dutch timezone (Europe/Amsterdam)"""
    if isinstance(dt, datetime):
        return dt.astimezone(DUTCH_TZ)
    return None


try:
    # Connect to the database
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    
    cursor = conn.cursor()
    cursor.execute("SET client_encoding TO 'UTF8';")

    # Query for extracting all the necessary fields from the database.
    query = """
    SELECT 
        b.id AS booking_id,
        b.customer,
        b.delivery_datetime,
        b.last_pickup,
        b.cargo_closing,
        b.cargo_opening,
        b.first_pickup,
        dl.code AS delivery_location_code,
        b.original ->> 'container_number' AS container_number,
        t.id AS transport_id,
        t.modality AS transport_modality,
        t.type AS transport_type,
        t.area AS transport_region,
        t.index AS transport_index,
        s.id AS stop_id,
        s.index AS stop_index,
        s.location AS stop_location_id,
        sl.code AS stop_location_code,
        s.estimated_arrival,
        s.estimated_departure,
        s.planned_arrival,
        s.planned_departure,
        s.actual_arrival,
        s.actual_departure,
        s.estimated_service_duration,
        s.planned_service_duration,
        s.actual_service_duration
    FROM bookings AS b
    LEFT JOIN transports AS t ON t.booking = b.id
    LEFT JOIN locations AS dl ON b.delivery_location = dl.id
    LEFT JOIN stops AS s ON s.transport = t.id
    LEFT JOIN locations AS sl ON s.location = sl.id
    WHERE b.customer IS NOT NULL
      AND b.deleted != TRUE
      AND b.delivery_datetime IS NOT NULL
      AND TO_CHAR(b.delivery_datetime, 'HH24:MI:SS') != '01:00:00' --Because +1 timezone
      AND TO_CHAR(b.delivery_datetime, 'HH24:MI:SS') != '02:00:00' --or +2 depending on daylight savings
      AND (t.deleted != TRUE)
      AND (s.deleted != TRUE)
      AND (cargo_opening IS NOT NULL OR cargo_closing IS NOT NULL OR  first_pickup IS NOT NULL OR last_pickup IS NOT NULL)
      AND (NOT t.type IN ('coupling', 'pancake', 'merged', 'transfer'))
      AND (t.area IN ('region', 'local'))
    ORDER BY customer, booking_id, transport_index, stop_index
     """

    cursor.execute(query)
    rows = cursor.fetchall()

    # First group all transports and stops by booking ID
    bookings_map = {}

    for row in rows:
        booking_id = row[0]
        transport_id = row[9]
        transport_type = row[11]
        transport_region = row[12]
        transport_index = row[13]
        
        # If booking not seen before, create entry
        if booking_id not in bookings_map:
            bookings_map[booking_id] = {
                "booking_id": booking_id,
                "customer": row[1],
                "delivery_datetime": convert_to_dutch_time(row[2]),
                "last_pickup": convert_to_dutch_time(row[3]),
                "cargo_closing": convert_to_dutch_time(row[4]),
                "cargo_opening": convert_to_dutch_time(row[5]),
                "first_pickup": convert_to_dutch_time(row[6]),
                "delivery_location": row[7],
                "container_number": row[8],
                "transports": {},
                "total_time": None
            }

        # Add transport if it exists and not already added
        if transport_id and transport_id not in bookings_map[booking_id]['transports']:
            bookings_map[booking_id]['transports'][transport_id] = {
                "transport_id": transport_id,
                "modality": row[10],
                "type": transport_type,
                "region": transport_region,
                "index": transport_index,
                "stops": []
            }

        # Add stop if it exists
        if row[14]:  # stop_id exists
            stop_details = {
                "stop_id": row[14],
                "index": row[15],
                "location_id": row[16],
                "location_code": row[17],
                "estimated_arrival": convert_to_dutch_time(row[18]),
                "estimated_departure": convert_to_dutch_time(row[19]),
                "planned_arrival": convert_to_dutch_time(row[20]),
                "planned_departure": convert_to_dutch_time(row[21]),
                "actual_arrival": convert_to_dutch_time(row[22]),
                "actual_departure": convert_to_dutch_time(row[23]),
                "estimated_service_time": row[24] if len(row) > 24 else None,
                "planned_service_time": row[25] if len(row) > 25 else None,
                "actual_service_time": row[26] if len(row) > 26 else None
            }
            if (row[15] != None):
                bookings_map[booking_id]['transports'][transport_id]['stops'].append(stop_details)

    # Prepare final result
    result = []
    for booking_id, booking in bookings_map.items():
        # Sort transports by their index
        transports_list = sorted(
            booking['transports'].values(),
            key=lambda x: x['index']
        )
        
        # Sort stops within each transport by their index
        all_stops = []
        for transport in transports_list:
            transport['stops'].sort(key=lambda x: x['index'])
            all_stops.extend(transport['stops'])

        # Append to the result
        result.append({
            "booking_id": booking['booking_id'],
            "customer": booking['customer'],
            "delivery_datetime": booking['delivery_datetime'],
            "last_pickup": booking['last_pickup'],
            "first_pickup": booking['first_pickup'],
            "cargo_opening": booking['cargo_opening'],
            "cargo_closing": booking['cargo_closing'],
            "delivery_location": booking['delivery_location'],
            "container_number": booking['container_number'],
            "transports_count": len(transports_list),
            "total_time": booking['total_time'],
            "transports_details": transports_list
        })

    # Save to JSON file
    with open('validationSetWithDatetime.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)

    print(f"Success! Processed {len(result)} bookings with estimated service times")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()

#Need to fix encoding to utf-8
def fix_encoding(input_file, output_file):
    print(f"Loading data...")
    """
    Reads a JSON file, fixes encoding issues, and saves it back.
    """
   
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = f.read()

    # Fix incorrect double encoding (if necessary)
    corrected_data = raw_data.encode("utf-8", "ignore").decode("utf-8")

   
    json_data = json.loads(corrected_data)

    # Save fixed JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

input_file = 'validationSetWithDatetime.json'  
fixed_file = 'fixed_encoding.json' 
output_file = 'output_with_date_column.json'  



fix_encoding(input_file, fixed_file)


# Delete the delivery datetime and stops for the validation set 

def validationDelete():
    validation_df = pd.read_json("fixed_encoding.json")

    # Set the 'delivery_datetime' column to null (None) for all rows
    validation_df['delivery_datetime'] = None

    # Remove the 'stops' for all transports in the 'transports_details' column
    for idx, row in validation_df.iterrows():
        if 'transports_details' in row:
            for transport in row['transports_details']:
                # Clear the 'stops' key for each transport
                transport['stops'] = []

    # Save the updated DataFrame to a new JSON file
    validation_df.to_json("validationSetWihoutDatetime.json", orient="records", lines=True)

validationDelete()

# Finds the average times for each route:
# E.G. route: CTT_COMPANY, avg_time: 140
# Or If route going back, COMPANY_CTT, avg_time: 90
# If decoupling, route is reversed

def find_route_averages(validation_set_path, route_averages_path, output_file_path, typeString):
    # Load the route averages data 
    route_averages_df = pd.read_json(route_averages_path, lines=True)
    route_averages_dict = {route["route"]: route for _, route in route_averages_df.iterrows()}
    
    # Result list to hold the found routes with their associated times
    found_routes = []
    

    validation_data_df = pd.read_json(validation_set_path)
  
    # Iterate over each record in the validation set dataframe
    for _, record in validation_data_df.iterrows():
       
        # Extract the transport details and booking id
        transports_details = record.get('transports_details', [])
        booking_id = record.get('booking_id')
        
        # Check if its the first two stop or the last two and extract the stop location codes
        if (typeString == 'firstTwo'):
            first_stop_location_code = transports_details[0]['stops'][0].get('location_code')
            second_stop_location_code = transports_details[0]['stops'][1].get('location_code')
        elif(typeString == 'secondTwo'):
            if(transports_details[0]['type'] == 'decoupling'):
                first_stop_location_code = transports_details[0]['stops'][1].get('location_code')
                second_stop_location_code = transports_details[0]['stops'][0].get('location_code')
            else:
                first_stop_location_code = transports_details[0]['stops'][1].get('location_code')
                second_stop_location_code = transports_details[0]['stops'][2].get('location_code')
        
        if first_stop_location_code and second_stop_location_code:
            
            # Create the route identifier in the format stop1_stop2
            route_id = f"{first_stop_location_code}_{second_stop_location_code}"
            
            # Look for the route in the route averages data
            if route_id in route_averages_dict:
                found_route = route_averages_dict[route_id]
                # Append the route and its associated average time in minutes to the result
                route_data = {
                    "booking_id": booking_id,
                    "average_time_minutes": found_route.get('average_time_minutes', 95)  # Default to 95 if not found
                }
                found_routes.append(route_data)

            else:
                # If the route is not found, assign a default average time
                route_data = {
                    "booking_id": booking_id,
                    "average_time_minutes":95  # Default time if route not found. 95 is average for all deliveries
                }
                found_routes.append(route_data)
        else:
                        
                # If the route is not found, assign a default average time
                route_data = {
                    "booking_id": booking_id,
                    "average_time_minutes": 95  # Default time if route not found
                }
                found_routes.append(route_data)


    # Save the found routes with their times to the output file as a JSON
    with open(output_file_path, 'w') as output_file:
        json.dump(found_routes, output_file, indent=4)
    
    print(f"Routes with times have been saved to {output_file_path}")

# Call the function for the first two and last two stops
validation_set_path = 'validationSetWithDatetime.json'
route_averages_path = 'route_averages_first_two.json'
output_file_path = 'found_routes_with_times_first_two.json'
typeString = 'firstTwo'

find_route_averages(validation_set_path, route_averages_path, output_file_path, typeString)

typeString = 'secondTwo'
validation_set_path = 'validationSetWithDatetime.json'
route_averages_path = 'route_averages_second_two.json'
output_file_path = 'found_routes_with_times_second_two.json'

find_route_averages(validation_set_path, route_averages_path, output_file_path, typeString)

# One hot encoding columns for data normalization

def add_week_day_hour_customer_location_columns(input_file, output_file):
    """
    Enhances the JSON data by adding:
    - One-hot encoding for each unique customer (`isCustomer_{customer_id}`) with 1 or 0
    - One-hot encoding for each unique location (`isLocation_{location_name}`) with 1 or 0
    - 52 week columns (`week_1` to `week_52`)
    - 365 day columns (`day_1` to `day_365`)
    - 24 hour columns (`ishour1` to `ishour24`)
    - Transports count column as an integer
    """

    # Load the data from the JSON file
    print("Loading data...")
    df = pd.read_json(input_file, lines=True)

    # Create 52 week columns
    week_columns = [f'week_{week}' for week in range(1, 53)]
    week_df = pd.DataFrame(0, index=df.index, columns=week_columns)

    # Create 365 day columns
    day_columns = [f'day_{day}' for day in range(1, 366)]
    day_df = pd.DataFrame(0, index=df.index, columns=day_columns)

    # Create 24 hour columns
    hour_columns = [f'ishour{hour}' for hour in range(1, 25)]
    hour_df = pd.DataFrame(0, index=df.index, columns=hour_columns)

    # One-hot encode 'customer' column (1 or 0)
    customer_dummies = pd.get_dummies(df['customer'], prefix='isCustomer', dtype=int) if 'customer' in df.columns else pd.DataFrame()

    # One-hot encode 'location_name' column (1 or 0)
    location_dummies = pd.get_dummies(df['delivery_location'], prefix='isLocation', dtype=int) if 'delivery_location' in df.columns else pd.DataFrame()

    #  Ensure transports_count is an integer column
    df['transports_count'] = df['transports_count'].fillna(0).astype(int)

    #  Same for total_time
    df['total_time'] = df['total_time'].fillna(0).astype(int)

    # Concat all new columns with the original DataFrame
    df = pd.concat([df, week_df, day_df, hour_df, customer_dummies, location_dummies], axis=1)

    # Iterate through each row and mark corresponding week, day, and hour columns with 1
 

    # Save the modified df to the output JSON file
    df.to_json(output_file, orient='records', lines=True)
    


input_file = 'validationSetWihoutDatetime.json'  
output_file = 'output_with_week_day_hour_customer_location.json'  

add_week_day_hour_customer_location_columns(input_file, output_file)


# Dropping unnecessary columns 

def delete_columns(input_file, output_file):
    # List of columns to remove
 
    columns_to_delete = ["cargo_opening","first_pickup","delivery_location","booking_id","customer","delivery_datetime","last_pickup","container_number", "transports_details","modality","date", "cargo_closing", "day_366"]    
    
    # Load the JSON file
    df = pd.read_json(input_file, lines=True)
    
    # Drop the specified columns
    df.drop(columns=columns_to_delete, errors='ignore', inplace=True)
    
    # Save the modified dframe to a new JSON file
    df.to_json(output_file, orient='records', lines=True)
    print(f"Modified JSON saved as '{output_file}'")


input_file = 'output_with_week_day_hour_customer_location.json'       
output_file = 'output_validation.json' 

delete_columns(input_file, output_file)

