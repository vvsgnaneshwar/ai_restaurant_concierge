# 🍽️ Nom Nom: The AI Concierge for GoodFoods

Nom Nom is a conversational AI agent designed to automate restaurant reservations and customer inquiries for the GoodFoods chain. This agent enhances customer experience while optimizing restaurant operations through intelligent scheduling, natural conversation, and data-driven insights.

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- A working internet connection (for LLM and embedding API calls)

### Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/vvsgnaneshwar/ai_restaurant_concierge.git
   cd ai_restaurant_concierge
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Environment Configuration:**
   Create a `.env` file in the root directory and add your API key:
   ```
   GEMINI_API_KEY=your_google_generative_ai_api_key
   ```

4. **Prepare Data:**
   Place the required Excel files inside the `data/` folder:
   - reservations.xlsx
   - tables_capacity.xlsx
   - venue_adjacency_matrices.xlsx
   - venues.xlsx

5. **Run the app:**
   ```
   streamlit run app.py
   ```

## 🧠 Prompt Engineering Approach

To ensure robust task management while working within context limitations, the following strategy was adopted:

### Initial Attempts with Function Calling Models
Native function-calling models like Gemini 1.5, LLaMA 3.1, and Qwen 3-8B showed context window limitations, failing when overloaded with instructions + long chat histories.

### Custom Prompt Format Inspired by Thinking Tags
A single LLM is prompted using:
- Concise, essential instructions
- A trimmed conversation history
- A directive to "think" before acting (enclosed in `<think>` tags)
- A structured output that clearly distinguishes between helper-agent and customer-facing responses

### Tools as Minimal Structured Calls
Tools are described simply in the system prompt, and invoked with readable function-like strings. This reduces token usage compared to large JSON schema.

### Helper Agent Loop
The LLM can request tools via function-like output. A helper agent interprets these, performs the operations, and appends the results to the context, completing the loop.

### Context Window Optimization
Tool descriptions, examples, and rarely-used paths are kept minimal. If retrieval is necessary, it's delegated to a dedicated agent with a free context window.

## 💬 Example Conversations (User Journeys)

### 1. New Reservation
![new_booking](https://github.com/user-attachments/assets/d93d0a56-a1d3-4f78-a9ba-a81a596ad51f)

### 2. Cancel Reservation
![cancellation](https://github.com/user-attachments/assets/dbbaf1ea-2d8a-4976-9484-0c3fdf287458)

### 3. Nearest Venues
![nearest_venues](https://github.com/user-attachments/assets/8a3c33cc-5ac8-470a-9569-eb9518788e9b)

### 4. Venue Recommendation
![recommendation](https://github.com/user-attachments/assets/f30ed23e-9ad8-43d3-b7ca-cdf62723bd26)

## 💼 Business Strategy Summary

Derived from the Use Case Document and integrated into this system's architecture:

### 📈 Goal
Automate 95% of booking and inquiry interactions, enabling:
- Reduced staff load
- Higher customer satisfaction
- Data-driven personalization

### 🎯 Use Case
GoodFoods aims to deploy Nom Nom across all its Hyderabad locations to manage:
- Reservations (new, view, modify, cancel)
- Nearby venue search
- Cuisine & venue details
- Personalized suggestions based on preferences

### 🚀 Phased Rollout Plan
1. Internal A/B Testing
2. Pilot Deployment for few Locations
3. Full Rollout + Voice Integration

## 💡 Key Innovations
- MILP-based intelligent table selection
- Venue recommendation using semantic similarity
- Operational insights-driven upsell suggestions
- Future-proof: ready for migration to RAG+SQL backend

## 📌 Assumptions, Limitations, and Enhancements

### ✅ Assumptions
- Table arrangements for a venue are static
- Reservation slot duration is fixed
- Randomized durations are used for proximity due to billing issues with Maps API

### ⚠️ Limitations
- No live sync between reservation and POS systems yet
- No OTP verification yet (required for security in production)

### 🔮 Future Enhancements
- Integrate Google Maps API for live ETA
- Dynamic slot length estimation via regression
- Expand from Excel-based KB to vector+SQL hybrid backend
- Voice assistant integration
- Multilingual support
- Feedback collection and loyalty loop
