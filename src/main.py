
# import libraries
import os
import math
from datetime import datetime, date
from typing import Dict
import yfinance as yf
import random
import string

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from dotenv import load_dotenv
load_dotenv()



# initialize llm
def initialize_llm(model_name="llama3.2:3b", temperature=0, max_tokens=256):
    """
    Initializes the ChatOllama model with specified parameters.
    Returns:
        ChatOllama: Instance of ChatOllama model.
    """
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print("Model initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing the model: {e}")
        return None

#/////////////////////////////////////////////////////////////////////  

# get current time
@tool
def get_current_time() -> str:
    """
    Returns the current time as a formatted string like 'HH:MM:SS AM/PM', in 12-hour format with AM/PM notation.
    
    Returns:
        str: A string representing the current time in the format 'HH:MM:SS AM/PM'.
    """
    try:
        # Get the current date and time
        now = datetime.now()

        # Format as a string in 12-hour format with AM/PM notation
        time_str = now.strftime("%I:%M:%S %p")

        # Return the formatted string
        return time_str
    except Exception as e:
        # Return an error message if an exception occurs
        return f"An error occurred while fetching the time: {e}"
    
#/////////////////////////////////////////////////////////////////////  

# get current date
@tool
def current_date() -> str:
    """
    Return the current date in YYYY-MM-DD format.
    
    Returns:
        str: A string representing the current date in the format 'YYYY-MM-DD'.
    """
    try:
        return date.today().strftime("%Y-%m-%d")
    except Exception as e:
        return f"Error occurred while fetching the date: {e}"

#/////////////////////////////////////////////////////////////////////

# calculate area
@tool   
def calculate_area(shape: str, dimensions: Dict[str, float]) -> float:
    """
    Calculate the area of a given shape based on its dimensions.

    Parameters:
    shape (str): The shape of the object. Can be "circle", "square", "rectangle", or "triangle".
    dimensions (dict[str, float]): A dictionary containing the necessary dimensions for the shape.
                       - For "circle", provide {'radius': value}.
                       - For "square", provide {'side': value}.
                       - For "rectangle", provide {'length': value, 'width': value}.
                       - For "triangle", provide {'base': value, 'height': value}.

    Returns:
    float: The area of the shape.

    Raises:
    ValueError: If an unsupported shape is provided, or if the necessary dimensions are missing.
    """
    try:
        if shape == "circle":
            if 'radius' not in dimensions:
                raise ValueError("Missing 'radius' for circle")
            return math.pi * (dimensions['radius'] ** 2)
        
        elif shape == "square":
            if 'side' not in dimensions:
                raise ValueError("Missing 'side' for square")
            return dimensions['side'] ** 2
        
        elif shape == "rectangle":
            if 'length' not in dimensions or 'width' not in dimensions:
                raise ValueError("Missing 'length' or 'width' for rectangle")
            return dimensions['length'] * dimensions['width']
        
        elif shape == "triangle":
            if 'base' not in dimensions or 'height' not in dimensions:
                raise ValueError("Missing 'base' or 'height' for triangle")
            return 0.5 * dimensions['base'] * dimensions['height']
        
        else:
            raise ValueError("Unsupported shape")
        
    except Exception as e:
        return f"Error occurred while calculating area: {e}"
#////////////////////////////////////////////////////////////////////

# perform math
@tool
def perform_math_operation(operation: str, operands: Dict[str, float]) -> float:
    """
    Perform a mathematical operation based on the given operands.

    Parameters:
    operation (str): The operation to perform. Can be one of the following:
                     'add', 'subtract', 'multiply', 'divide', 'square', 'sqrt'.
    operands (dict[str, float]): A dictionary containing the operands for the operation.
                               - For 'add', provide {'a': value, 'b': value}.
                               - For 'subtract', provide {'a': value, 'b': value}.
                               - For 'multiply', provide {'a': value, 'b': value}.
                               - For 'divide', provide {'a': value, 'b': value}.
                               - For 'square', provide {'x': value}.
                               - For 'sqrt', provide {'x': value}.
    
    Returns:
    float: The result of the operation.

    Raises:
    ValueError: If an unsupported operation is provided or incorrect operands are given.
    ZeroDivisionError: If division by zero is attempted.
    """
    
    try:
        if operation == "add":
            # Ensure 'a' and 'b' are provided
            return operands['a'] + operands['b']
        
        elif operation == "subtract":
            # Ensure 'a' and 'b' are provided
            return operands['a'] - operands['b']
        
        elif operation == "multiply":
            # Ensure 'a' and 'b' are provided
            return operands['a'] * operands['b']
        
        elif operation == "divide":
            # Ensure 'a' and 'b' are provided and check for division by zero
            if operands['b'] == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return operands['a'] / operands['b']
        
        elif operation == "square":
            # Ensure 'x' is provided
            return operands['x'] ** 2
        
        elif operation == "sqrt":
            # Ensure 'x' is provided and check for negative numbers
            if operands['x'] < 0:
                raise ValueError("Cannot calculate square root of a negative number")
            return math.sqrt(operands['x'])
        
        else:
            raise ValueError("Unsupported operation")
    
    except KeyError as e:
        raise ValueError(f"Missing operand: {e}")

#////////////////////////////////////////////////////////////////////

# temparature converter
@tool
def temperature_converter(value: float, to_scale: str) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Parameters:
        value (float): The temperature value to convert.
        to_scale (str): The target scale, either 'C' for Celsius or 'F' for Fahrenheit.
        
    Returns:
        float: The converted temperature.
        
    Raises:
        ValueError: If an invalid target scale is provided.
        TypeError: If the provided value is not a float or int.
    """
    try:
        # Check if the target scale is valid
        if to_scale.lower() == 'f':
            # Celsius to Fahrenheit
            return (value * 9/5) + 32
        elif to_scale.lower() == 'c':
            # Fahrenheit to Celsius
            return (value - 32) * 5/9
        else:
            raise ValueError("Invalid target scale. Use 'C' for Celsius or 'F' for Fahrenheit.")
    except ValueError as e:
        raise ValueError(f"Temperature conversion error: {e}")

#////////////////////////////////////////////////////////////////////

# BMI calculator
@tool
def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate the Body Mass Index (BMI).
    
    Parameters:
        weight (float): Weight in kilograms.
        height (float): Height in meters.
        
    Returns:
        float: The calculated BMI.
        
    Raises:
        ValueError: If height is zero or negative, or if weight or height is not a float.
        
    Example:
        calculate_bmi(65, 1.8)  # BMI for a person with weight 65kg and height 1.8m
        calculate_bmi(60, 1.7)  # BMI for a person with weight 60kg and height 1.7m
    """
    try:
        # Check for valid height
        if height <= 0 and weight <= 0:
            raise ValueError("Height and weight must be greater than zero.")
        
        # Calculate BMI
        bmi = weight / (height ** 2)
        return bmi

    except TypeError:
        raise TypeError("Weight and height must be numbers.")
    
#////////////////////////////////////////////////////////////////////

# password generator
@tool
def suggest_password(length: int = 8) -> str:
    """
    Generate a random password that meets specified security requirements.

    The password will:
    - Be at least 8 characters long (or the specified length, whichever is greater)
    - Contain at least one lowercase letter (a-z)
    - Contain at least one uppercase letter (A-Z)
    - Contain at least one digit (0-9)
    - Contain at least one special character (e.g., !@#$%^&*)

    Parameters:
    length (int): Desired length of the password (minimum is 8). Defaults to 8.

    Returns:
    str: A randomly generated password that meets the specified criteria.

    Example:
    >>> suggest_password(12)
    'A3b!dF2#gH1k'
    """
    try:
        # Check if length is a positive integer
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer.")

        # Ensure password meets minimum length of 8
        if length < 8:
            length = 8

        # Define character pools
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*"

        # Make sure the password includes at least one of each required type
        password = [
            random.choice(lower),
            random.choice(upper),
            random.choice(digits),
            random.choice(special)
        ]

        # Fill the rest of the password length with random choices from all pools
        all_characters = lower + upper + digits + special
        password += random.choices(all_characters, k=length - 4)

        # Shuffle the password to randomize order
        random.shuffle(password)

        # Join list into a string and return
        return ''.join(password)

    except TypeError:
        raise TypeError("The length parameter must be an integer.")
    except ValueError as ve:
        raise ValueError(f"Invalid length value: {ve}")

#////////////////////////////////////////////////////////////////////

# wikipedia search
@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for the provided query like about any topics (person, place, thing, etc) and return the result. 

    Parameters:
        query (str): The search term or query to look up on Wikipedia.

    Returns:
        str: The summary or result of the Wikipedia query, or an error message if the query fails.
    """
    try:
        # Initialize the Wikipedia API wrapper
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        # Run the query and return the result
        return wikipedia.run(query)

    except ConnectionError:
        return "Network error: Unable to connect to Wikipedia. Please check your internet connection."
    except Exception as e:
        return f"An error occurred: {str(e)}"

#////////////////////////////////////////////////////////////////////

# weather forecast  
@tool
def fetch_weather(city: str) -> dict:
    """
    Fetches the current weather data for a specified city using the OpenWeatherMap API.

    Parameters:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        dict: A dictionary containing the weather data for the specified city if the request is successful.
              If an error occurs (e.g., network issues, invalid API key, unrecognized city), 
              returns a dictionary with an "error" key and a corresponding error message.
    """
    try:
        # Initialize the OpenWeatherMap API wrapper
        weather = OpenWeatherMapAPIWrapper()
        
        # Fetch the current weather data for the specified city
        weather_data = weather.run(city)
        
        # Return the weather data if fetched successfully
        return weather_data
    
    except ConnectionError:
        return "Network error: Unable to connect to Wikipedia. Please check your internet connection."
    except Exception as e:
        return f"An error occurred: {str(e)}"  

#////////////////////////////////////////////////////////////////////

# get stock price
@tool   
def get_stock_price_and_currency(ticker_symbol: str) -> str:
    """
    Fetches the latest stock price and currency used in the market for a given ticker symbol from yfinance.

    Parameters:
    ticker_symbol (str): The ticker symbol of the stock for which the information is to be retrieved.

    Returns:
    str: A string representing the current stock price followed by the currency used in the market, 
         formatted as "<price> <currency>."
    
    Example:
    >>> get_stock_price_and_currency('AAPL')
    '175.96 USD.'
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        stock_price = ticker.info['currentPrice']  # Latest market price
        currency = ticker.info['currency']         # Currency used in the market

        price = f"{stock_price} {currency}."
        return price

    except Exception as e:
        return f"Error: {str(e)}"

#////////////////////////////////////////////////////////////////////

tools = [get_current_time,
         get_stock_price_and_currency,
         calculate_area,
         perform_math_operation,
         fetch_weather,
         temperature_converter,
         calculate_bmi,
         current_date,
         suggest_password,
         search_wikipedia,
            ]

# Initialize the ChatOllama model 
llm = initialize_llm()   

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

parser = StrOutputParser()

tools_map  = {
    "get_current_time": get_current_time,
    "fetch_weather": fetch_weather,
    "current_date": current_date,
    "search_wikipedia": search_wikipedia,
    "get_stock_price_and_currency": get_stock_price_and_currency,
    "calculate_area": calculate_area,
    "perform_math_operation": perform_math_operation,
    "temperature_converter": temperature_converter,
    "calculate_bmi": calculate_bmi,
    "suggest_password": suggest_password,
}


def chat_interaction(query):
    messages = [SystemMessage("You are a helpful assistant"), HumanMessage(query)]
    # Call the model without 'await' since it's a synchronous function
    
    try:
        ai_msg = llm_with_tools.invoke(messages)
        
        # Check if ai_msg is empty or has no tool calls
        if not ai_msg or not ai_msg.tool_calls:
            print("Warning: No valid tool calls generated by the model.")
            fallback_msg = "Sorry, I couldn't generate a valid response or find the correct tool to execute your request."
            messages.append(HumanMessage(fallback_msg))
        else:
            print("Tool Calls:", ai_msg.tool_calls)
            messages.append(ai_msg)

            for tool_call in ai_msg.tool_calls:
                selected_tool = tools_map.get(tool_call["name"].lower())

                if selected_tool:
                    tool_msg = selected_tool.invoke(tool_call)
                    messages.append(tool_msg)
                else:
                    print(f"Warning: Tool '{tool_call['name']}' not found in tools map. Using fallback.")
                    # Fallback mechanism for missing tools
                    fallback_msg = "I couldn't find the correct tool to execute your request."
                    messages.append(HumanMessage(fallback_msg))
        
        # Final response after all tool calls
        result = llm_with_tools.invoke(messages)
        assistant_response = parser.invoke(result)
        return assistant_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."

import asyncio

async def main():
    print("Starting LLM Function calling...")
    while True:
        user_input = input("\nask => ")
        if user_input.lower() == "exit":
            print("Exiting chat interaction.")
            break

        # Call chat_interaction  within the loop
        print("AI => ", chat_interaction(user_input))

if __name__ == "__main__":
    asyncio.run(main())

