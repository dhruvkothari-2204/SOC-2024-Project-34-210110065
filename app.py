import streamlit as st
import spacy
import yfinance as yf
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Initialize Spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize LLaMA model
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b")

st.title("Investment Strategy Chatbot")
user_input = st.text_area("Enter your investment preferences:")
def extract_investment_params(text):
    doc = nlp(text)
    investment_params = {
        "goal": None,
        "risk_tolerance": None,
        "amount": None,
        "horizon": None,
        "sectors": None,
        "volatility_tolerance": None
    }
    
    # Example extraction logic (can be expanded)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            investment_params["amount"] = ent.text
        elif ent.label_ == "DATE":
            investment_params["horizon"] = ent.text
        # Add more conditions as needed
    
    return investment_params

params = extract_investment_params(user_input)
def set_default_params(params):
    if not params["goal"]:
        params["goal"] = "medium-term"
    if not params["risk_tolerance"]:
        params["risk_tolerance"] = "medium"
    if not params["amount"]:
        params["amount"] = "10000"
    if not params["horizon"]:
        params["horizon"] = "1 year"
    if not params["sectors"]:
        params["sectors"] = ["technology", "healthcare"]
    if not params["volatility_tolerance"]:
        params["volatility_tolerance"] = "medium"
    return params

params = set_default_params(params)
def fetch_top_stocks(sectors):
    if not sectors:
        sectors = ["technology", "healthcare"]
    
    # Placeholder for fetching top stocks logic
    stocks = yf.Tickers(" ".join(sectors)).tickers
    return stocks[:10]

stocks = fetch_top_stocks(params["sectors"])
def predict_stock_prices(stocks, horizon):
    # Placeholder prediction logic
    predictions = {stock: "predicted price" for stock in stocks}
    return predictions

predictions = predict_stock_prices(stocks, params["horizon"])
def calculate_volatility(stocks):
    volatilities = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1y")
        volatilities[stock] = hist['Close'].pct_change().std()
    return volatilities

volatilities = calculate_volatility(stocks)
def analyze_stocks(stocks, predictions, volatilities, volatility_tolerance):
    filtered_stocks = []
    for stock in stocks:
        if volatilities[stock] < volatility_tolerance:
            filtered_stocks.append(stock)
    return filtered_stocks

analyzed_stocks = analyze_stocks(stocks, predictions, volatilities, params["volatility_tolerance"])
def summarize_analysis(stocks):
    summary = f"Based on your preferences, we recommend the following stocks: {', '.join(stocks)}."
    return summary

summary = summarize_analysis(analyzed_stocks)
def generate_advice(summary):
    inputs = tokenizer(summary, return_tensors="pt")
    outputs = model.generate(**inputs)
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return advice

advice = generate_advice(summary)
st.write(advice)
if st.button("Get Investment Advice"):
    params = extract_investment_params(user_input)
    params = set_default_params(params)
    stocks = fetch_top_stocks(params["sectors"])
    predictions = predict_stock_prices(stocks, params["horizon"])
    volatilities = calculate_volatility(stocks)
    analyzed_stocks = analyze_stocks(stocks, predictions, volatilities, params["volatility_tolerance"])
    summary = summarize_analysis(analyzed_stocks)
    advice = generate_advice(summary)
    st.write(advice)
