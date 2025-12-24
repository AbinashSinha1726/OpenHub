import os
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from serpapi import GoogleSearch
from crewai import Agent, Task, Crew, Process, LLM
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================
#  Initialize Open AI (LLM)
# ==========================
@lru_cache(maxsize=1)
def initialize_llm():
    """Initialingand caching LLM instance to avoid repeated initializations."""
    return LLM(
        model="gpt-5.1",
        api_key=OPENAI_API_KEY
    )

# ==========================
#  Pydantic Models
# ==========================
class FlightRequest(BaseModel):
    origin: str
    destination: str
    outbound_date: str
    return_date: str

class HotelRequest(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str

class ItineraryRequest(BaseModel):
    destination: str
    check_in_date: str
    check_out_date: str
    flights: str
    hotels: str

class FlightInfo(BaseModel):
    airline: str
    price: str
    duration: str
    stops: str
    departure: str
    arrival: str
    travel_class: str
    return_date: str
    airline_logo: str

class HotelInfo(BaseModel):
    name: str
    price: str
    rating: float
    location: str
    link: str

class AIResponse(BaseModel):
    flights: List[FlightInfo] = []
    hotels: List[HotelInfo] = []
    ai_flight_recommendation: str = ""
    ai_hotel_recommendation: str = ""
    itinerary: str = ""

# ==========================
#  Initialize FastAPI
# ==========================
app = FastAPI(title="Travel Planning API", version="1.1.0")

# ==========================
#  Fetch Data from SerpAPI
# ==========================
async def run_search(params):
    """Generic function to run SerpAPI searches asynchronously."""
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
    except Exception as e:
        logger.exception(f"SerpAPI search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")

async def search_flights(flight_request: FlightRequest):
    """Fetch real-time flight details from Google using SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} to {flight_request.destination}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": flight_request.origin.strip().upper(),
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "INR"
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Flight search error: {search_results['error']}")
        return {"error": search_results["error"]}

    best_flights = search_results.get("best_flights", [])
    if not best_flights:
        logger.warning("No flights found in search results")
        return []

    formatted_flights = []
    for flight in best_flights:
        if not flight.get("flights") or len(flight["flights"]) == 0:
            continue

        first_leg = flight["flights"][0]
        formatted_flights.append(FlightInfo(
            airline=first_leg.get("airline", "Unknown Airline"),
            price=str(flight.get("price", "N/A")),
            duration=f"{flight.get('total_duration', 'N/A')} min",
            stops="Nonstop" if len(flight["flights"]) == 1 else f"{len(flight['flights']) - 1} stop(s)",
            departure=f"{first_leg.get('departure_airport', {}).get('name', 'Unknown')} ({first_leg.get('departure_airport', {}).get('id', '???')}) at {first_leg.get('departure_airport', {}).get('time', 'N/A')}",
            arrival=f"{first_leg.get('arrival_airport', {}).get('name', 'Unknown')} ({first_leg.get('arrival_airport', {}).get('id', '???')}) at {first_leg.get('arrival_airport', {}).get('time', 'N/A')}",
            travel_class=first_leg.get("travel_class", "Economy"),
            return_date=flight_request.return_date,
            airline_logo=first_leg.get("airline_logo", "")
        ))

    logger.info(f"Found {len(formatted_flights)} flights")
    return formatted_flights


async def search_hotels(hotel_request: HotelRequest):
    """Fetch hotel information from SerpAPI."""
    logger.info(f"Searching hotels for: {hotel_request.location}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "en",
        "gl": "us",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "INR",
        "sort_by": 3,
        "rating": 8
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Hotel search error: {search_results['error']}")
        return {"error": search_results["error"]}

    hotel_properties = search_results.get("properties", [])
    if not hotel_properties:
        logger.warning("No hotels found in search results")
        return []

    formatted_hotels = []
    for hotel in hotel_properties:
        try:
            formatted_hotels.append(HotelInfo(
                name=hotel.get("name", "Unknown Hotel"),
                price=hotel.get("rate_per_night", {}).get("lowest", "N/A"),
                rating=hotel.get("overall_rating", 0.0),
                location=hotel.get("location", "N/A"),
                link=hotel.get("link", "N/A")
            ))
        except Exception as e:
            logger.warning(f"Error formatting hotel data: {str(e)}")

    logger.info(f"Found {len(formatted_hotels)} hotels")
    return formatted_hotels

# ==============================================
#  Format Data for AI
# ==============================================
def format_travel_data(data_type, data):
    """Generic formatter for both flight and hotel data."""
    if not data:
        return f"No {data_type} available."

    if data_type == "flights":
        formatted_text = "**Available flight options**:\n\n"
        for i, flight in enumerate(data):
            formatted_text += (
                f"**Flight {i + 1}:**\n"
                f"✈️ **Airline:** {flight.airline}\n"
                f" ₹ **Price:** ${flight.price}\n"
                f"⏱️ **Duration:** {flight.duration}\n"
                f"🛑 **Stops:** {flight.stops}\n"
                f"🕔 **Departure:** {flight.departure}\n"
                f"🕖 **Arrival:** {flight.arrival}\n"
                f"💺 **Class:** {flight.travel_class}\n\n"
            )
    elif data_type == "hotels":
        formatted_text = "**Available Hotel Options**:\n\n"
        for i, hotel in enumerate(data):
            formatted_text += (
                f"**Hotel {i + 1}:**\n"
                f"🏨 **Name:** {hotel.name}\n"
                f" ₹ **Price:** ₹{hotel.price}\n"
                f"⭐ **Rating:** {hotel.rating}\n"
                f"📍 **Location:** {hotel.location}\n"
                f"🔗 **More Info:** [Link]({hotel.link})\n\n"
            )
    else:
        return "Invalid data type."

    return formatted_text.strip()


# =======================
#  AI Analysis Functions
# =======================
async def get_ai_recommendation(data_type, formatted_data):
    """Unified function for getting AI recommendations for both flights and hotels."""
    logger.info(f"Getting {data_type} analysis from AI")
    llm_model = initialize_llm()

    # Agent Configuration based on data type
    if data_type == "flights":
        role = "AI Flight Analyst"
        goal = "Recommend the best flight by assessing price, duration, stops, and convenience."
        backstory = f"An AI specialist that performs detailed comparisons of flight options across multiple criteria."
        description = """
        Based on the information below, evaluate the available flights and recommend the optimal option.:

        **Recommendation Summary:**
        - ** ₹ Price:** Provide a thorough justification for why this flight is the most cost-effective and convenient choice.
        - **⏱️ Duration:** Provide an analysis showing why this flight has a superior total travel time relative to alternatives.
        - **🛑 Stops:** Describe how this flight minimizes layovers while maintaining overall efficiency..
        - **💺 Travel Class:** Provide a detailed assessment of this flight’s comfort features and amenities, highlighting why they outperform alternatives..

        Use the provided flight data as the basis for your recommendation. Be sure to justify your choice using clear reasoning for each attribute. Do not repeat the flight details in your response.
        """
    elif data_type == "hotels":
        role = "AI Hotel Analyst"
        goal = "Analyze hotel options and recommend the best one by considering price, rating, location and amenities."
        backstory = f"AI expert which provides in-depth analysis in comparing hotel options based on multiple factors."
        description = """
        Using the analysis below, recommend the best hotel with a detailed explanation considering price, rating, location, and amenities.

        **AI Hotel Recommendation**
        Based on the analysis below, we recommend the top hotel option::

        **Recommendation Summary:**:
        - **₹ Price:** This hotel represents the most cost-effective option, providing excellent amenities and services relative to its price.
        - **⭐ Rating:** The hotel’s higher rating reflects consistently positive reviews and a higher level of service quality. Compared to alternatives, this indicates a better overall guest experience, making it the optimal selection.
        - **📍 Location: Strategically located near major points of interest, the hotel offers excellent convenience for travelers.
        - **🏨 Amenities: With offerings such as high-speed Wi-Fi, a pool, fitness facilities, and free breakfast, the hotel meets diverse traveler needs. These amenities improve convenience, relaxation, and productivity, making it suitable for families, solo travelers, and business guests alike.

        📝 **Reasoning Requirements**:
        - Each section should provide a clear rationale demonstrating why this hotel is optimal, considering key factors such as price, rating, location, and amenities.
        - Conduct a comparison with the other available options and highlight the factors that make this one the standout choice.
        - Provide well-organized justification to make the recommendation transparent and easy to understand.
        - Ensure the recommendation incorporates multiple criteria so the traveler can weigh all relevant aspects before deciding.
        """
    else:
        raise ValueError("Invalid data type for AI recommendation")

    analyze_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_model,
        verbose=False
    )

    analyze_task = Task(
        description=f"{description}\n\nData to analyze:\n{formatted_data}",
        agent=analyze_agent,
        expected_output=f"A concise, data-driven recommendation highlighting the top {data_type} selection according to the analyzed details."
    )

    analyst_crew = Crew(
        agents=[analyze_agent],
        tasks=[analyze_task],
        process=Process.sequential,
        verbose=False
    )

    try:
        crew_results = await asyncio.to_thread(analyst_crew.kickoff)
        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get(role, f"No {data_type} recommendation available.")
        else:
            return str(crew_results)
    except Exception as e:
        logger.exception(f"Error in AI {data_type} analysis: {str(e)}")
        return f"Unable to generate {data_type} recommendation due to an error."


async def generate_itinerary(destination, flights_text, hotels_text, check_in_date, check_out_date):
    """Generate a detailed travel itinerary based on flight and hotel information."""
    try:
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")

        days = (check_out - check_in).days

        llm_model = initialize_llm()

        analyze_agent = Agent(
            role="AI Travel Planner",
            goal="Generate a full itinerary for the traveler, incorporating both flight schedules and hotel accommodations.",
            backstory="AI-driven itinerary planner offering an optimized daily plan including travel logistics, accommodation, and key experiences.",
            llm=llm_model,
            verbose=False
        )

        analyze_task = Task(
            description=f"""
            Based on the following details, create a {days}-day itinerary for the user:

            **Flight Details**:
            {flights_text}

            **Hotel Details**:
            {hotels_text}

            **Destination**: {destination}

            **Travel Dates**: {check_in_date} to {check_out_date} ({days} days)

            The itinerary should include:
            - Flight Details ✈️  
                        Arrival and departure times
                        Flight numbers and airlines
                        Duration and layovers
            - Hotel Information 🏨 
                        Check-in and check-out times
                        Hotel name, rating, and location
                        Key amenities
            - Day-by-Day Activities 📅 
                        Morning, afternoon, and evening plans
                        Estimated durations for each activity
                        Flexibility for leisure or optional events
            - Must-Visit Attractions 🏛️ 
                        Top landmarks or experiences
                        Suggested visit times and duration
                        Tips for avoiding crowds or optimizing time
            - Restaurant Recommendations 🍴 
                        Breakfast, lunch, and dinner options
                        Specialty cuisine or local favorites
                        Approximate price range
            - Local Transportation Tips 🚌🚇 
                        Best modes of transport between destinations
                        Estimated travel time
                        Cost-saving or convenient options

             **Itinerary Formatting Guidelines**:
            -Headings: 
                    # for the main itinerary title 
                    ## for each day,
                    ### for sub-sections like Flights, Hotel, Activities
            -Emojis: Use relevant emojis for quick visual cues: 
                    🏛️ Landmarks / attractions
                    🍽️ Restaurants / meals
                    🏨 Hotel stays
                    ✈️ Flights / travel
            -Bullet Points:
                    Use - or * to list activities, restaurants, or attractions
            -Estimated Timings:
                    Include approximate start and end times for activities (e.g., 09:00 AM – 11:00 AM)
            -Visual Appeal:
                    Keep sections clearly separated
                    Use bold for key points (hotel name, flight numbers, restaurant names)
                    Maintain consistent formatting for easy readability
            """,
            agent=analyze_agent,
            expected_output="A comprehensive, Markdown-formatted itinerary including flights, accommodations, and a detailed daily schedule, enhanced with emojis, headers, and bullet points for readability."
        )

        itinerary_planner_crew = Crew(
            agents=[analyze_agent],
            tasks=[analyze_task],
            process=Process.sequential,
            verbose=False
        )

        crew_results = await asyncio.to_thread(itinerary_planner_crew.kickoff)

        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get("AI Travel Planner", "No itinerary available.")
        else:
            return str(crew_results)

    except Exception as e:
        logger.exception(f"Error generating itinerary: {str(e)}")
        return "Unable to generate itinerary due to an error. Please try again later."


# ===============
#  API Endpoints
# ===============
@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(flight_request: FlightRequest):
    """Search flights and get AI recommendation."""
    try:
        flights = await search_flights(flight_request)

        if isinstance(flights, dict) and "error" in flights:
            raise HTTPException(status_code=400, detail=flights["error"])

        if not flights:
            raise HTTPException(status_code=404, detail="No flights found")

        flights_text = format_travel_data("flights", flights)

        ai_recommendation = await get_ai_recommendation("flights", flights_text)

        return AIResponse(
            flights=flights,
            ai_flight_recommendation=ai_recommendation
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Flight search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flight search error: {str(e)}")

@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(hotel_request: HotelRequest):
    """Search hotels and get AI recommendation."""
    try:
        hotels = await search_hotels(hotel_request)

        if isinstance(hotels, dict) and "error" in hotels:
            raise HTTPException(status_code=400, detail=hotels["error"])

        if not hotels:
            raise HTTPException(status_code=404, detail="No hotels found")

        hotels_text = format_travel_data("hotels", hotels)

        ai_recommendation = await get_ai_recommendation("hotels", hotels_text)

        return AIResponse(
            hotels=hotels,
            ai_hotel_recommendation=ai_recommendation
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Hotel search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hotel search error: {str(e)}")


@app.post("/complete_search/", response_model=AIResponse)
async def complete_travel_search(flight_request: FlightRequest, hotel_request: Optional[HotelRequest] = None):
    """Search for flights and hotels concurrently and get AI recommendations for both."""
    try:
        if hotel_request is None:
            hotel_request = HotelRequest(
                location=flight_request.destination,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        flight_task = asyncio.create_task(get_flight_recommendations(flight_request))
        hotel_task = asyncio.create_task(get_hotel_recommendations(hotel_request))

        flight_results, hotel_results = await asyncio.gather(flight_task, hotel_task, return_exceptions=True)

        if isinstance(flight_results, Exception):
            logger.error(f"Flight search failed: {str(flight_results)}")
            flight_results = AIResponse(flights=[], ai_flight_recommendation="Could not retrieve flights.")

        if isinstance(hotel_results, Exception):
            logger.error(f"Hotel search failed: {str(hotel_results)}")
            hotel_results = AIResponse(hotels=[], ai_hotel_recommendation="Could not retrieve hotels.")

        flights_text = format_travel_data("flights", flight_results.flights)
        hotels_text = format_travel_data("hotels", hotel_results.hotels)

        itinerary = ""
        if flight_results.flights and hotel_results.hotels:
            itinerary = await generate_itinerary(
                destination=flight_request.destination,
                flights_text=flights_text,
                hotels_text=hotels_text,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        return AIResponse(
            flights=flight_results.flights,
            hotels=hotel_results.hotels,
            ai_flight_recommendation=flight_results.ai_flight_recommendation,
            ai_hotel_recommendation=hotel_results.ai_hotel_recommendation,
            itinerary=itinerary
        )
    except Exception as e:
        logger.exception(f"Complete travel search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Travel search error: {str(e)}")


@app.post("/generate_itinerary/", response_model=AIResponse)
async def get_itinerary(itinerary_request: ItineraryRequest):
    """Generate an itinerary based on provided flight and hotel information."""
    try:
        itinerary = await generate_itinerary(
            destination=itinerary_request.destination,
            flights_text=itinerary_request.flights,
            hotels_text=itinerary_request.hotels,
            check_in_date=itinerary_request.check_in_date,
            check_out_date=itinerary_request.check_out_date
        )

        return AIResponse(itinerary=itinerary)
    except Exception as e:
        logger.exception(f"Itinerary generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Itinerary generation error: {str(e)}")


# ===================
# Run FastAPI Server
# ===================
if __name__ == "__main__":
    logger.info("Starting Travel Planning API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)