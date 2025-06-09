import dotenv
import os
import re
from google import genai
from google.genai import types
import ast
from datetime import datetime, date
import calendar
import pulp
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import google.generativeai as gen_ai
from sklearn.metrics.pairwise import cosine_similarity
import random

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

# gets the llm response to helper agent or customer
def get_assistant_response(conversation_history):

    prompt =  f'''
    SYSTEM MESSAGE:
    You are Nom Nom, GoodFoods' AI assistant who helps customers with their reservations and queries.
    The Helper Agent can use python tools/functions. When provided with the necessary parameters, the Helper Agent provides you the necessary inputs which can be used to continue the conversation.

    CONTEXT:
    GoodFoods is a growing restaurant chain with 20 venues across the Hyderabad city.
    The only 20 venues are Malkajgiri, Secunderabad, Bowenpally, Padma Rao Nagar, Rampally, Imax, Begumpet, Uppal, RTC X Roads, Himayat Nagar, Masab Tank, Kompally,
    Charminar, Gudimalkapur, Hitech City, SR Nagar, Vanasthalipuram, Gachibowli, Jubilee Hills and Manikonda.
    In general, GoodFoods offers multiple cuisines such as Indian, Chinese, Italian, Continental and many more.
    FYI, today is {datetime.now().strftime("%d-%m-%Y")} (DD-MM-YYYY), {datetime.now().strftime("%A")}. The time now is {datetime.now().strftime("%H:%M")}.
    Number of days in this month is {calendar.monthrange(datetime.now().year, datetime.now().month)[1]}.

    TOOLS AVAILABLE:
    1) new_booking(name, phone_number, venue, guest_size, date, time)
    2) fetch_bookings(name, phone_number): fetches all the bookings for the given name and phone number.
    3) cancel_booking(name, phone_number, date, time)
    4) fetch_about_venue(venue): fetches the general cuisine, specialities, website and phone number of the venue. detailed menu & photos are present in the website.
    5) fetch_nearest_venues(customer_location): returns two nearest venues to the customer_location with durations.
    6) fetch_venues_with_description(description): fetches the top 2 relevant venues based on the description. description is a single string which can have natural language. it can also have customer preferences. venues cannot be completly relevant to the description.

    Parameter formats: name - str, phone_number - int (10 digits), venue/customer_location - str, guest_size - int, date - str (DD-MM-YYYY) , time - str (HH:MM, 24 hour format).
    Venue is a location like "Malkajgiri", "Secunderabad", etc from the above.

    Think out loudly before responding. While outputting, enclose all your thoughts within the tags <think> and </think>.

    The final output should have one of the following formats exactly:

    1) If there is no need to use the tools, then just continue the conversation with the customer:
    <think> Your thoughts </think>
    {{"helper_agent": "not_required", "assistant_response_to_customer": "Thanks for visiting us."}}

    helper_agent should be set to not_required if you are responding directly to the customer.

    OR

    2) If there is a need to use tools, give the functions with the all the required parameters in content to be used by the Helper Agent:
    <think> Your thoughts </think>
    {{"helper_agent": "required", "assistant_response_to_helper_agent": ["new_booking("xyz", 9383839993, "sjhjhdf", 3, "DD-MM-YYYY", "HH:MM")", "fetch_booking("xyz", 8979399324)", "fetch_about_venue("Malkajgiri")"]}}

    Before using the tools, you need to always ensure that you have all the required parameters. If there are missing parameter, ask the customer to provide them.
    Helper Agent replies are not visible to the customer.

    IMPORTANT:
    1) For modification, you can use fetch_bookings to get the current bookings. After clarifying with the customer for date and time, you can cancel the existing booking and make a new booking.
    2) Do not trouble the customer with formats. You need to convert the parameters to the required formats before using them. When replying to customer, use natural formats for date and time (AM or PM).
    3) Think step by step logically.
    4) Remember to check if you have all the required parameters before using the tools. Especially name, customer_location if needed.
    5) While using venue as parameter, use the closest match with the above mentioned venues as to avoid spelling mistakes.
    6) While suggesting venues based on description, suggest the venue or venues only if they match with the description or conversation context. Filter the fetched venues based on the customer's preferences.

    CONVERSATION HISTORY:'''
    prompt  = "\n".join([prompt, conversation_history])

    response = client.models.generate_content(
        model="gemini-1.5-flash-8b", 
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1
        ),
    ).text

    pattern = r'{"helper_agent":\s*[^}]*}'

    match = re.search(pattern, response)
    start_index = match.start()
    end_index = match.end()

    # extract the dict from the response
    response = ast.literal_eval(response[start_index:end_index])

    print(response)

    return response

def get_reply_to_customer(conversation_history, customer_msg):

    customer_msg_str = f"Customer to Assistant: {customer_msg}"
    conversation_history = "\n".join([conversation_history, customer_msg_str])

    assistant_response = get_assistant_response(conversation_history)
    assitant_msg = ""

    while assistant_response["helper_agent"] == "required": 
        helper_agent_response = "Helper Agent to Assistant: "
        
        for function_call in assistant_response['assistant_response_to_helper_agent']:
            helper_agent_response = helper_agent_response + str(eval(function_call))

        assitant_msg = f"Assistant to Helper Agent: Execute these functions: {assistant_response['assistant_response_to_helper_agent']}."
        conversation_history = "\n".join([conversation_history, assitant_msg, helper_agent_response])
        assistant_response = get_assistant_response(conversation_history)

    # update conversation history
    assistant_response_str = f"Assistant to Customer: {assistant_response['assistant_response_to_customer']}"
    conversation_history = "\n".join([conversation_history, assistant_response_str])

    return assistant_response['assistant_response_to_customer'], conversation_history

def solve_table_selection(adjacency_matrix, capacity_table, N, w_capacity=10, w_num_tables=60, w_adjacency=30, w_distribution=1):
    # Extract table information
    table_numbers = list(adjacency_matrix.index)
    n_tables = len(table_numbers)
    
    # Create capacity dictionary
    capacity_dict = dict(zip(capacity_table['table_number'], capacity_table['seating_capacity']))
    
    # Create the problem
    prob = pulp.LpProblem("Restaurant_Table_Selection", pulp.LpMinimize)
    
    # Decision variables
    # x[i] = 1 if table i is selected, 0 otherwise
    x = {}
    for i in table_numbers:
        x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')
    
    # X[i,j] = 1 if both tables i and j are selected, 0 otherwise
    X = {}
    table_pairs = list(combinations(table_numbers, 2))
    for i, j in table_pairs:
        X[(i, j)] = pulp.LpVariable(f"X_{i}_{j}", cat='Binary')
    
    # Calculate adjacency score
    adjacency_score = 0
    for i, j in table_pairs:
        M_ij = adjacency_matrix.loc[i, j]
        adjacency_score += M_ij * X[(i, j)]
    
    # Calculate distribution score
    distribution_score = 0
    for i, j in table_pairs:
        Ci = capacity_dict[i]
        Cj = capacity_dict[j]
        # For absolute value |Ci - Cj|, we use the actual difference since it's constant
        distribution_score += abs(Ci - Cj) * X[(i, j)]
    
    # Calculate total capacity
    total_capacity = 0
    for i in table_numbers:
        Ci = capacity_dict[i]
        total_capacity += Ci * x[i]

    # Calculate total number of tables
    total_num_tables = 0
    for i in table_numbers:
        total_num_tables += x[i]
    
    # Objective function
    objective = (w_capacity * (total_capacity - N) 
                + w_num_tables * total_num_tables 
                - w_adjacency * adjacency_score 
                + w_distribution * distribution_score)
    
    prob += objective

    eps = 0.001

    # Constraints for X[i,j] variables
    for i, j in table_pairs:
        # X[i,j] >= x[i] + x[j] - 1
        prob += X[(i, j)] >= x[i] + x[j] - 1
        
        # X[i,j] <= 1 - eps * (2 - (x[i] + x[j]))
        prob += X[(i, j)] <= 1 - eps * (2 - (x[i] + x[j]))
    
    # Constraint: Total capacity should be at least N
    prob += total_capacity >= N
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output
    
    # Extract results
    selected_tables = []
    table_details = []
    
    for i in table_numbers:
        if x[i].varValue == 1:
            selected_tables.append(i)
            table_details.append({
                'table_number': i,
                'capacity': capacity_dict[i]
            })

    return {
        'status': pulp.LpStatus[prob.status],
        'selected_tables': selected_tables,
        'table_details': table_details
    }

def new_booking(name, phone_number, venue, guest_size, date, time):
    try:
        # --- Input Validation ---
        if not all([name, phone_number, venue, date, time]):
            return "Missing required booking information. Please check the inputs."

        if not isinstance(guest_size, int) or guest_size <= 0:
            return "Invalid guest size. Please provide a positive number."

        slot_time = 60  # minutes

        def time_in_minutes(time_str):
            return int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])

        # --- Load Data ---
        reservations_table = pd.read_excel("data/reservations.xlsx")
        tables = pd.read_excel("data/tables_capacity.xlsx")

        reservations = reservations_table[(reservations_table["venue"] == venue) & (reservations_table["date"] == date)].copy()
        reservations['time'] = reservations['time'].apply(lambda x: time_in_minutes(x))
        tables = tables[tables["venue"] == venue]

        # --- Filter Reserved Tables ---
        tables_reserved = list(reservations[
            (reservations["time"] >= time_in_minutes(time) - slot_time) &
            (reservations["time"] <= time_in_minutes(time) + slot_time)
        ]['table_number'].unique())

        # --- Load Adjacency Matrix ---
        adjacency_matrix = pd.read_excel("data/venue_adjacency_matrices.xlsx", sheet_name=venue, index_col=0)
        adjacency_matrix.columns = adjacency_matrix.columns.astype(int)
        adjacency_matrix.index = adjacency_matrix.index.astype(int)
        adjacency_matrix = adjacency_matrix.drop(index=tables_reserved, columns=tables_reserved)

        # --- Check Available Capacity ---
        capacity_table = tables[~tables["table_number"].isin(tables_reserved)][["table_number", "seating_capacity"]]
        total_capacity = sum(capacity_table['seating_capacity'])

        if total_capacity < guest_size:
            return "Restaurant does not have enough seating capacity for the guests at the desired time. Apologize to the customer. Ask the customer for another time."

        # --- Solve Table Allocation ---
        table_selection_results = solve_table_selection(
            adjacency_matrix=adjacency_matrix,
            capacity_table=capacity_table,
            N=guest_size
        )

        # --- Prepare New Reservation Rows ---
        rows = []
        for table in table_selection_results['table_details']:
            rows.append([name, int(phone_number), venue, guest_size, date, time, table['table_number'], table['capacity']])

        # --- Update Reservation Sheet ---
        reservations_table = pd.read_excel("data/reservations.xlsx")
        new_reservations = pd.DataFrame(rows, columns=reservations_table.columns)
        reservations_table = pd.concat([reservations_table, new_reservations], ignore_index=True)
        reservations_table.to_excel("data/reservations_updated.xlsx", index=False)

        return "Reservation Successful!"

    except Exception as e:
        return "An unexpected error occurred while processing the booking. Please check if the inputs are valid."

def fetch_bookings(name, phone_number):
    try:
        # --- Input Validation ---
        if not name or not phone_number:
            return "Missing name or phone number. Please provide both to fetch the booking."        

        # --- Load Reservations ---
        reservations_table = pd.read_excel("data/reservations.xlsx")
        
        # --- Filter by Name and Phone Number ---
        reservations = reservations_table[
            (reservations_table["name"] == name) &
            (reservations_table["phone_number"] == phone_number)
        ]

        if reservations.empty:
            return f"No booking found for the given name: {name} and phone number: {phone_number}."

        # --- Group by Booking ---
        grouped = reservations.groupby(
            ['venue', 'guest_size', 'date', 'time']
        )['table_number'].apply(list).reset_index()

        # --- Format Response ---
        response_lines = [f"{len(grouped)} booking{'s' if len(grouped) > 1 else ''} found."]
        
        for i, row in grouped.iterrows():
            time_24 = row['time']
            try:
                time_12 = datetime.strptime(time_24, "%H:%M").strftime("%I:%M %p")
            except ValueError:
                time_12 = time_24  # Fallback in case of unexpected format

            response_lines.append(
                f"Booking {i+1}: venue - {row['venue']}, guest size - {row['guest_size']}, "
                f"date - {row['date']}, time - {time_12}, table numbers reserved - {row['table_number']}."
            )

        return " ".join(response_lines)

    except Exception:
        return "An error occurred while retrieving the booking details. Please check if the inputs are valid."

def cancel_booking(name, phone_number, date, time):
    try:
        # --- Input Validation ---
        if not all([name, phone_number, date, time]):
            return "Missing booking details. Please provide name, phone number, date, and time to cancel the booking."

        # --- Load Reservations ---
        file_path = "data/reservations.xlsx"
        reservations_table = pd.read_excel(file_path)

        # --- Identify Bookings to Cancel ---
        mask = (
            (reservations_table["name"] == name) &
            (reservations_table["phone_number"] == phone_number) &
            (reservations_table["date"] == date) &
            (reservations_table["time"] == time)
        )

        bookings_to_cancel = reservations_table[mask]

        if bookings_to_cancel.empty:
            return f"No booking found for {name} at {date} {time}. Please check if the inputs are valid."

        # --- Remove Matching Bookings ---
        updated_reservations = reservations_table[~mask]

        # --- Save Updated Table ---
        updated_reservations.to_excel("data/reservations_updated.xlsx", index=False)

        return f"Booking for {name} on {date} at {time} has been successfully cancelled."

    except Exception:
        return "An error occurred while attempting to cancel the booking. Please check if the inputs are valid."

def fetch_about_venue(venue):
    try:
        # --- Input Validation ---
        if not venue:
            return "Please provide a venue name to fetch its information."

        # --- Load Venues Data ---
        venues_table = pd.read_excel("data/venues.xlsx")
        print(venues_table.head())

        # --- Filter for the Venue ---
        venue_info = venues_table[venues_table["venue"] == venue]

        if venue_info.empty:
            return f"No information found for venue: {venue}."

        # --- Extract Details ---
        about = venue_info.iloc[0].get("about", "No description available.")
        website = venue_info.iloc[0].get("website", "No website listed.")
        phone = venue_info.iloc[0].get("phone_number", "No phone number available.")
        print('xyz!!!!!!!!!!!')
        return f"About GoodFoods, {venue}: {about} Website: {website}. Phone Number: {phone}."

    except Exception as e:
        print(e)
        return "An error occurred while retrieving venue information. Please check if the inputs are valid."

def fetch_nearest_venues(customer_location):
    try:
        # --- Input Validation ---
        if not customer_location:
            return "Please provide your location to find the nearest venues."

        # --- Load Venues List ---
        venues_df = pd.read_excel("data/venues.xlsx")

        # --- Simulate Durations (in minutes) ---
        venues_df["duration_min"] = [random.randint(10, 120) for _ in range(len(venues_df))]

        # --- Find Two Nearest Venues ---
        nearest = venues_df.sort_values(by="duration_min").head(2)

        # --- Format Response ---
        response_lines = [f"Nearest venues to {customer_location}:"]
        for i, row in nearest.iterrows():
            response_lines.append(f"{row['venue']} - {row['duration_min']} min")

        return " ".join(response_lines)

    except Exception:
        return "An error occurred while fetching nearest venues. Please try again later."

def fetch_venues_with_description(description: str) -> str:
    try:
        # --- Input validation ---
        if not description or not description.strip():
            return "Please provide a description to find relevant venues."

        EMBEDDING_MODEL = "models/text-embedding-004"

        # 1. Generate an embedding for the customer's preference query.
        query_embedding = gen_ai.embed_content(
            model=EMBEDDING_MODEL,
            content=description,
            task_type="clustering"
        )["embedding"]

        # 2. Read venue information
        venues = pd.read_excel("data/venues.xlsx")
        if venues.empty:
            return "Venue data is currently unavailable."

        # 3. Generate embeddings for venue descriptions
        venue_embeddings = gen_ai.embed_content(
            model=EMBEDDING_MODEL,
            content=venues["about"].tolist(),
            task_type="clustering"
        )["embedding"]

        # 4. Compute cosine similarities
        similarities = cosine_similarity([query_embedding], venue_embeddings)[0]

        # 5. Find indices of top 3 venues
        top_indices = np.argsort(similarities)[-2:][::-1]

        # 6. Prepare venue info for LLM prompt
        venue_lines = []
        for i, idx in enumerate(top_indices, 1):
            venue_name = venues.iloc[idx]['venue']
            about_text = venues.iloc[idx]['about']
            venue_lines.append(f"{i}. {venue_name}: {about_text}")

        venues_info = "\n".join(venue_lines)

        # 7. Create LLM prompt
        prompt = f'''Venues Info:
        {venues_info}
        Description: {description}
        Based on the description and venues info, output which venue or venues are the best fit for the customer along with a separate justification per venue. Output has to be short and a single line without any line breaks.
        '''

        # 8. Call Gemini
        response = client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1),
        ).text

        return response

    except Exception as e:
        return f"An error occurred while searching for venues. Please try again later."