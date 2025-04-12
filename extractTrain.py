import psycopg2
import json
import pytz
from datetime import datetime, timedelta
import pandas as pd
import pickle


pd.set_option('future.no_silent_downcasting', True)

# Database connection details (fill in the ones you have)
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
    FROM bookings_archive AS b
    LEFT JOIN transports_archive AS t ON t.booking = b.id
    LEFT JOIN locations AS dl ON b.delivery_location = dl.id
    LEFT JOIN stops_archive AS s ON s.transport = t.id
    LEFT JOIN locations AS sl ON s.location = sl.id
    WHERE b.customer IS NOT NULL
      AND b.deleted != TRUE
      AND b.delivery_datetime IS NOT NULL
      AND TO_CHAR(b.delivery_datetime, 'HH24:MI:SS') != '01:00:00' --Because +1 Timezone
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
            # Drop stop if has no location
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
    with open('processed_bookings_full.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)

    print(f"Success! Processed {len(result)} bookings with estimated service times")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()

# Fixing utf-8 encoding
def fix_encoding(input_file, output_file):
    """
    Reads a JSON file, fixes encoding issues, and saves it back.
    """
  
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = f.read()

  
    corrected_data = raw_data.encode("utf-8", "ignore").decode("utf-8")


    json_data = json.loads(corrected_data)

    # Save fixed JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


#Adding a column 'date'  which is the delivery_datetime, makes check if not null 

def add_date_column(input_file, output_file):
    print("Loading data...")
    """
    Reads a JSON file, assigns a date per row (delivery_datetime), 
    removes rows without a valid date, and saves the modified data.
    """

    df = pd.read_json(input_file, encoding="utf-8")

    # Assign the delivery_datetime as the primary date
    df['date'] = df['delivery_datetime']

    
    df = df[df['date'].notna()]

    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Save the modified DataFrame to JSON
    df.to_json(output_file, orient='records', lines=True, date_format='iso')

input_file = 'processed_bookings_full.json' 
fixed_file = 'fixed_encoding.json' 
output_file = 'output_with_date_column.json'  

# Fix encoding issues
fix_encoding(input_file, fixed_file)


add_date_column(fixed_file, output_file)

# One hot encoding columns for data normalizations

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
    
    df = pd.read_json(input_file, lines=True)

   
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle timezone conversion (UTC → Amsterdam time)
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert('Europe/Amsterdam')
    else:
        df['date'] = df['date'].dt.tz_convert('Europe/Amsterdam')

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

    # Ensure transports_count is an integer column
    df['transports_count'] = df['transports_count'].fillna(0).astype(int)

    # Same for total_time

    df['total_time'] = df['total_time'].fillna(0).astype(int)

    # Concat all new columns with the original dframe
    df = pd.concat([df, week_df, day_df, hour_df, customer_dummies, location_dummies], axis=1)

    # Iterate through each row and mark corresponding week, day, and hour columns with 1
    for index, row in df.iterrows():
        if pd.notna(row['date']):  # Ensure the date is valid
            # Mark the corresponding week column
            week_num = row['date'].isocalendar()[1]
            df.at[index, f'week_{week_num}'] = 1  

            # Mark the corresponding day column
            day_of_year = row['date'].dayofyear
            df.at[index, f'day_{day_of_year}'] = 1  

            # Mark the corresponding hour column
            hour = row['date'].hour  
            df.at[index, f'ishour{hour + 1}'] = 1  

    # Save the modified DataFrame to the output JSON file
    df.to_json(output_file, orient='records', lines=True)

input_file = 'output_with_date_column.json'  
output_file = 'output.json'  

add_week_day_hour_customer_location_columns(input_file, output_file)




# Delete unnecessary columns. 
def delete_columns(input_file, output_file):
    # List of columns to remove
    # columns_to_delete = ['id', 'delivery_datetime', 'last_pickup', 
    #                      'cargo_closing','date']
    
    columns_to_delete = ["cargo_opening","first_pickup","delivery_location","booking_id","customer","delivery_datetime","last_pickup","container_number", "transports_details","modality","date", "cargo_closing", "day_366"]    
    
    # Load the JSON file (assuming newline-delimited JSON)
    df = pd.read_json(input_file, lines=True)
    
    # Drop the specified columns; errors='ignore' ensures no error if a column is missing
    df.drop(columns=columns_to_delete, errors='ignore', inplace=True)
    
    # Save the modified DataFrame to a new JSON file
    df.to_json(output_file, orient='records', lines=True)
   

    customer_columns = [col for col in df.columns if "customer" in col.lower()]
    customer_data = df[customer_columns]

    location_columns = [col for col in df.columns if "location" in col.lower()]
    location_data = df[location_columns]

    # Save all the columns for customer and location to a py readable format
    with open("customer_columns.pkl", "wb") as f:
        pickle.dump(customer_data, f)
    with open("location_columns.pkl", "wb") as f:
        pickle.dump(location_data, f)

    print(f"Modified JSON saved as '{output_file}'")


input_file = 'output.json'    
output_file = 'fullDataSet.json'

delete_columns(input_file, output_file)

with open("processed_bookings_full.json", "r") as file:
    processed_bookings = json.load(file)


print("calculating route times...")

# Used for calculating difference between arrival and departure between two stops

def calculate_temp_time(first_stop, second_stop):
    """Calculate the time it took from the first stop to the second stop in minutes"""
    # Get arrival time at first stop, prefer actual, then planned, then estimated
    service_time = first_stop.get('actual_arrival') or first_stop.get('planned_arrival') or first_stop.get('estimated_arrival')
    # Get departure time at second stop, prefer actual, then planned, then estimated
    actual_departure_second_stop = second_stop.get('actual_departure') or second_stop.get('planned_departure') or second_stop.get('estimated_departure')

    if service_time and actual_departure_second_stop:
        # Convert the times to pandas datetime objects (this handles timezones automatically)
        service_time = pd.to_datetime(service_time)
        actual_departure_second_stop = pd.to_datetime(actual_departure_second_stop)

        # Calculate the time difference in minutes
        temp_time = (actual_departure_second_stop - service_time).total_seconds() / 60  # Convert seconds to minutes
        return temp_time
    return 0  # Return 0 if we can't calculate

# Appending two stops to a booking

def appendStops(booking, temp_time, temp_times, stop_1, stop_2):
        temp_times.append({
        "booking_id": booking["booking_id"],
        "stop_1_id": stop_1["stop_id"],
        "stop_1_location_code": stop_1["location_code"],
        "stop_2_id": stop_2["stop_id"],
        "stop_2_location_code": stop_2["location_code"],
        "temp_time_minutes": temp_time
        
    })
        return temp_times


def calculate_routes():
    # Initialize a list to store results for temp times
    temp_times_first_two = []
    temp_times_second_two = []
    # Initialize a dictionary to store route times for calculating averages
    route_times_first_two = {}
    route_times_second_two = {}

    # Iterate over the processed bookings dataset
    for booking in processed_bookings:
        transports = booking.get("transports_details", [])
        
        # Iterate over each transport and get the stop and type of it
        for transport in transports:
            stops = transport.get("stops", [])
            transport_type = transport.get( "type")
            
            # Ensure there are at least two stops (index 0 and 1) to calculate the time
            if len(stops) > 1:

                # First get a list for estimated route times for the first two stops
                stop_1 = stops[0]
                stop_2 = stops[1]
                
                
                # Calculate the time between Stop 1 and Stop 2
                temp_time = calculate_temp_time(stop_1, stop_2)
                if temp_time < 0: 
                    temp_time = 95
                
                # Append the temp time and route information
                temp_times_first_two = appendStops(booking=booking, temp_time=temp_time, temp_times=temp_times_first_two, stop_1=stop_1, stop_2= stop_2)

                # Create a route identifier as "stop1_stop2"
                route = f"{stop_1['location_code']}_{stop_2['location_code']}"

                # Add the temp time to the route dictionary for averaging
                if route not in route_times_first_two:
                    route_times_first_two[route] = []
                route_times_first_two[route].append(temp_time)

                # Estimated route times for going back
                # If decoupling, reverses stop_1 and stop_2 otherwise takes alst two stops

                if  transport_type == "decoupling":
                    temp_times_second_two = appendStops(booking=booking, temp_time=temp_time, temp_times=temp_times_second_two, stop_1=stop_2, stop_2= stop_1)

                    # Some stops have # need to remove it
                    stop_2 = (stop_2.get('location_code') or '').strip().replace('#', '')

                    route = f"{stop_2}_{stop_1['location_code']}"

                # Add the temp time to the route dictionary for averaging
                    if route not in route_times_second_two:
                        route_times_second_two[route] = []
                    route_times_second_two[route].append(temp_time)
                    
                else:
                    stop_1 = stops[1]
                    stop_2 = stops[2]
                    temp_time = calculate_temp_time(stop_1, stop_2)

                    # Same as before but for last two stops
                    if temp_time < 0: 
                        temp_time = 95
                    temp_times_second_two = appendStops(booking=booking, temp_time=temp_time, temp_times=temp_times_second_two, stop_1=stop_1, stop_2= stop_2)

                    stop_1 = (stop_1.get('location_code') or '').strip().replace('#', '')
                    route = f"{stop_1}_{stop_2['location_code']}"

                    # Add the temp time to the route dictionary for averaging
                    if route not in route_times_second_two:
                        route_times_second_two[route] = []
                    route_times_second_two[route].append(temp_time)
                  

    # Convert temp times results to a DF
    temp_times_first_two_df = pd.DataFrame(temp_times_first_two)

    temp_times_second_two_df = pd.DataFrame(temp_times_second_two)

    # Calculate the average time for each route 
    route_averages_first_two = []
    for route, times in route_times_first_two.items():
        average_time = sum(times) / len(times)  # Calculate the average
        route_averages_first_two.append({
            "route": route,
            "average_time_minutes": average_time
        })
    # Calculate the average time for each route for the last two stops
    route_averages_second_two = []
    for route, times in route_times_second_two.items():
        average_time = sum(times) / len(times)  # Calculate the average
        route_averages_second_two.append({
            "route": route,
            "average_time_minutes": average_time
        })
    # Convert route averages to a DataFrame
    route_averages_df_one = pd.DataFrame(route_averages_first_two)
    route_averages_df_two = pd.DataFrame(route_averages_second_two)

    
    # Save everything to JSON format

    temp_times_first_two_df.to_json("temp_times.json", orient="records", lines=True)
    route_averages_df_one.to_json("route_averages_first_two.json", orient="records", lines=True)



    temp_times_second_two_df.to_json("temp_times_second_two.json", orient="records", lines=True)
    route_averages_df_two.to_json("route_averages_second_two.json", orient="records", lines=True)

    # Calculate the avg time for going forward and back
    total_average_time = temp_times_first_two_df["temp_time_minutes"].mean()
    print(f"Total average time across all routes: {total_average_time:.2f} minutes")

calculate_routes()