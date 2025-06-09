Nom Nom: AI Restaurant Concierge 🍔
Nom Nom is a sophisticated, AI-powered conversational agent designed to automate reservation management and customer inquiries for the GoodFoods restaurant chain. It serves as the first point of digital contact, enhancing operational efficiency and elevating the customer experience.
This project leverages Google's Gemini Pro model within a robust tool-using agentic framework. The agent can handle a variety of tasks, from booking and canceling reservations to providing intelligent venue recommendations based on customer preferences.
![alt text](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
  (Replace with your deployed app URL)
🌟 Key Features
Core Reservation Functions:
New Reservation: Secure a new booking, gathering all necessary details (name, phone, venue, etc.).
View Reservations: Retrieve existing booking details for a customer.
Modify/Cancel Reservation: Seamlessly update or cancel existing bookings.
Intelligent Table Assignment: Utilizes a Mixed-Integer Linear Programming (MILP) model to find the optimal table configuration for any party size, maximizing seating capacity and ensuring guest comfort.
Information & Recommendation:
Venue Locator: Finds the nearest GoodFoods locations based on a customer's location.
Venue & Cuisine Information (RAG): Provides detailed information on menus, specialties, and venue ambiance using a Retrieval-Augmented Generation approach on a knowledge base.
⚙️ Tech Stack
Frontend: Streamlit
LLM: Google Gemini 1.5 Flash
Core Logic: Python
Optimization: PuLP (for the MILP model)
Data Handling: Pandas, NumPy
Embeddings & Similarity: google-generativeai, scikit-learn
🚀 Setup & Installation
Follow these steps to get the Nom Nom agent running on your local machine.
1. Clone the Repository
git clone https://github.com/your-username/nom-nom-agent.git
cd nom-nom-agent
Use code with caution.
Bash
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
macOS / Linux:
python3 -m venv venv
source venv/bin/activate
Use code with caution.
Bash
Windows:
python -m venv venv
.\venv\Scripts\activate
Use code with caution.
Bash
3. Install Dependencies
Install all the required Python packages from the requirements.txt file.
pip install -r requirements.txt
Use code with caution.
Bash
4. Set Up Environment Variables
The application requires a Google Gemini API key. Create a .env file in the root directory of the project:
touch .env
Use code with caution.
Add your API key to the .env file:
GEMINI_API_KEY="your_google_gemini_api_key_here"
Use code with caution.
Env
5. Prepare Data Files
The agent relies on several Excel files for its knowledge base. Ensure you have a data/ directory with the following files:
data/reservations.xlsx: Stores all booking records.
Columns: name, phone_number, venue, guest_size, date, time, table_number, capacity
data/tables_capacity.xlsx: Contains the seating capacity for each table at every venue.
Columns: venue, table_number, seating_capacity
data/venues.xlsx: A knowledge base with details about each restaurant location.
Columns: venue, about, website, phone_number
data/venue_adjacency_matrices.xlsx: An Excel file where each sheet is named after a venue and contains the adjacency matrix for its table layout. This is used by the MILP model to optimize table selection.
6. Run the Streamlit App
You are now ready to launch the application!
streamlit run app.py
Use code with caution.
Bash
Open your web browser and navigate to http://localhost:8501.
🧠 Prompt Engineering Approach
The reliability of this agent hinges on a carefully crafted prompt engineering strategy that guides the LLM to function as a dependable tool-using agent.
Initially, I experimented with models that support native function calling (like Llama 3.1, Gemini 1.5, and Qwen 3). However, I observed that these models could struggle with accuracy when presented with a long conversation history combined with extensive instructions and complex tool schemas.
To overcome this, I developed a more robust, explicit approach inspired by the "think-aloud" reasoning seen in models like Qwen.
Forced "Think-Aloud" Reasoning: The core prompt instructs the LLM to first reason about the user's request and its own plan within <think> tags. This forces the model to perform step-by-step logical deduction before generating a final, structured output. Since LLMs are next-token predictors, this intermediate reasoning step significantly improves the quality and reliability of the subsequent action.
Simplified Tool-Calling Schema: Instead of relying on large, context-consuming JSON schemas for tool definitions, the prompt provides the LLM with simple Python function signatures. This conserves precious context window space and is easier for the model to parse.
Structured JSON for Actions: The model is mandated to produce one of two exact JSON outputs:
Direct Response: When no tools are needed, it responds directly to the user.
{"helper_agent": "not_required", "assistant_response_to_customer": "How can I help you today?"}
Use code with caution.
Json
Tool Call: When a tool is required, it specifies the function call(s) for the helper agent.
{"helper_agent": "required", "assistant_response_to_helper_agent": ["new_booking(...)"]}
Use code with caution.
Json
Decoupled RAG: For complex, context-heavy tasks like venue recommendations based on semantic descriptions, the RAG logic is handled by a separate function (fetch_venues_with_description). This function independently generates embeddings and calls the LLM with a focused prompt, preventing the main conversation context from becoming bloated.
This hybrid approach creates a highly reliable agent that can handle complex conversational flows, accurately use tools, and manage its context window efficiently.
💬 Example Conversations
Here are a few examples of how users can interact with Nom Nom:
<details>
<summary><strong>Journey 1: Making a New Reservation</strong></summary>
👤 User: I'd like to book a table for 4 people at the Hitech City location for this Saturday at 8 PM.
🤖 Nom Nom: I can certainly help with that! Could you please provide your full name and a 10-digit phone number to finalize the booking?
👤 User: My name is Alex and my number is 1234567890.
🤖 Nom Nom: (Thinks and calls the new_booking tool...)
🤖 Nom Nom: Great! Your reservation for 4 guests at GoodFoods, Hitech City, on [Date of Saturday] at 08:00 PM is confirmed. We look forward to seeing you!
</details>
<details>
<summary><strong>Journey 2: Canceling a Reservation</strong></summary>
👤 User: I need to cancel a reservation.
🤖 Nom Nom: I'm sorry to hear that. To proceed, could you please provide the name, phone number, date, and time used for the booking?
👤 User: It's under Alex, number 1234567890, for this Saturday at 8 PM.
🤖 Nom Nom: (Thinks and calls the cancel_booking tool...)
🤖 Nom Nom: Thank you, Alex. I have successfully located and cancelled your booking for this Saturday at 08:00 PM. We hope to welcome you back another time!
</details>
<details>
<summary><strong>Journey 3: Finding a Nearby Venue</strong></summary>
👤 User: What are the closest locations to Jubilee Hills?
🤖 Nom Nom: (Thinks and calls the fetch_nearest_venues tool...)
🤖 Nom Nom: The nearest venues to Jubilee Hills are: Gachibowli (approx. 18 min away) and Manikonda (approx. 25 min away).
</details>
<details>
<summary><strong>Journey 4: Getting a Venue Recommendation (RAG)</strong></summary>
👤 User: I'm looking for a good place for a romantic dinner, something with a nice ambiance.
🤖 Nom Nom: (Thinks and calls the fetch_venues_with_description tool...)
🤖 Nom Nom: Based on your preference for a romantic dinner with a nice ambiance, I recommend GoodFoods, Jubilee Hills, which is known for its upscale decor and serene atmosphere perfect for special occasions. Another great option is GoodFoods, Gachibowli, offering a sophisticated setting with beautiful city views.
</details>
📈 Business Strategy Summary
This project is not just a technical implementation; it's a strategic solution to core business challenges in the restaurant industry.
Problem: High staff workload from repetitive calls, revenue loss from no-shows, and underutilized capacity during off-peak hours.
AI-Powered Solution:
Automate Core Tasks: Frees up staff to focus on high-value, in-person guest services.
Proactive Engagement: The architecture can be extended to send automated reminders and confirmations, reducing no-show rates.
Dynamic Incentives: The system can identify low-demand slots and proactively offer incentives (e.g., a complimentary appetizer) to drive traffic.
Key Success Metrics:
Interaction Success Rate: >95% of user goals completed without manual intervention.
Staff Efficiency Gain: >20% decrease in time spent by staff on phone-based tasks.
Channel Shift & Growth: >20% increase in reservations made through the AI agent.
Unique Competitive Advantages:
Proactive Operational Intelligence: The agent doesn't just book—it optimizes. The MILP model for table assignment is a key differentiator that improves seating efficiency.
End-to-End Customer Lifecycle Management: The system is designed to manage the entire customer journey, from pre-arrival reminders to post-dining feedback collection, creating a continuous loop for service improvement.
Intelligent Resource Optimization: The custom MILP model provides a tangible operational advantage by maximizing seating efficiency, especially for large parties—a common pain point that off-the-shelf solutions handle poorly.
🤝 Contributing
