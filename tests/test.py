import pytest
from src.main import (
    calculate_area,
    perform_math_operation,
    temperature_converter,
    calculate_bmi,
    suggest_password,
    get_current_time,
    current_date,
    search_wikipedia,
    fetch_weather,
    get_stock_price_and_currency,
    initialize_llm  )
from src.main import SystemMessage, HumanMessage 

tools = [calculate_area,
    perform_math_operation,
    temperature_converter,
    calculate_bmi,
    suggest_password,
    get_current_time,
    current_date,
    search_wikipedia,
    fetch_weather,
    get_stock_price_and_currency,
    initialize_llm
    ]
# Initialize LLM with tools
llm = initialize_llm()
llm_with_tools = llm.bind_tools(tools)


# Test case for LLM tool invocation with varied queries
@pytest.mark.parametrize("query, expected_tool", [
    ("What's the current time?", "get_current_time"),
    ("Tell me the stock price for AAPL", "get_stock_price_and_currency"),
    ("Calculate area of a circle with radius 5", "calculate_area"),
    ("sum of 5 and 7", "perform_math_operation"),
    ("What's the weather in Paris?", "fetch_weather"),
    ("Convert 100 Celsius to Fahrenheit", "temperature_converter"),
    ("Calculate BMI for 70 kg and 1.75 m", "calculate_bmi"),
    ("Today's date?", "current_date"),
    ("Generate a password of 10 characters", "suggest_password"),
    ("Who is Alan Turing?", "search_wikipedia")
])
def test_llm_tool_invocation(query, expected_tool):
    messages = [SystemMessage("You are a helpful assistant"), HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    assert ai_msg.tool_calls[0]['name'] == expected_tool
