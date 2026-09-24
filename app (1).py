import os
import uvicorn
import requests
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Culinary Tools ---
@tool
def search_restaurants(cuisine: str) -> str:
    """Search for top restaurants by cuisine type."""
    database = {
        "indian": "Peshawri (Mughlai), Karavalli (South Indian), Saravana Bhavan (Vegetarian)",
        "italian": "Toscano, Olive Bar & Kitchen, La Loca Maria",
        "chinese": "Yauatcha, Mainland China, Royal China",
        "street food": "Elco Chaat Center, Bademiya, Chandni Chowk Stalls"
    }
    return database.get(cuisine.lower(), "No restaurants found for that cuisine standard.")

@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Get current rating, price tier, and operating status for a restaurant."""
    mock_ratings = {
        "peshawri": {"rating": 4.8, "price_range": "₹₹₹₹", "status": "Open Now"},
        "toscano": {"rating": 4.5, "price_range": "₹₹₹", "status": "Open Now"},
        "yauatcha": {"rating": 4.6, "price_range": "₹₹₹₹", "status": "Closed - Opens at 7 PM"},
        "elco chaat center": {"rating": 4.3, "price_range": "₹", "status": "Open Now"}
    }
    
    key = restaurant_name.lower().strip()
    if key in mock_ratings:
        info = mock_ratings[key]
        info["restaurant"] = restaurant_name
        return json.dumps(info)
    
    return json.dumps({"restaurant": restaurant_name, "rating": 4.2, "price_range": "₹₹", "status": "Open Now"})

tools = [search_restaurants, get_restaurant_rating]

# --- 2. Initialize Model & Guardrailed Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to food, cuisines, and restaurant information. "
        "For any other roles, topics, questions, or general knowledge outside of food and restaurants, "
        "you must say exactly: 'I am not authorized to answer questions outside of food and restaurants.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the food and restaurant agent")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(title="Real-Time Food & Restaurant Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
