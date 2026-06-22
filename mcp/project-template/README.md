# MCP Weather Assistant Project

## 🎯 Project Goal

This project demonstrates a **Model Context Protocol (MCP)** server implementation with **Retrieval-Augmented Generation (RAG)** capabilities. It provides an intelligent weather assistant that:

- **Retrieves real-time weather data** from weather websites using Playwright web automation
- **Supports multiple locations** - USA and Israel weather forecasts
- **Integrates with Claude AI** - Uses Anthropic's Claude model for intelligent responses
- **Implements RAG** - The LLM bases responses on actual retrieved weather data, not just training data
- **Multi-tool architecture** - Separates concerns with dedicated weather scrapers and MCP servers

## 📁 Project Structure

```
project-template/
├── host.py                 # Main host that manages MCP clients and Claude integration
├── client.py               # MCP client for connecting to weather servers
├── weather_Israel.py       # MCP server for Israeli weather forecasts (Playwright-based)
├── weather_USA.py          # MCP server for USA weather forecasts (Playwright-based)
├── pyproject.toml          # Project dependencies and configuration
└── README.md               # This file
```

## 🔧 Key Components

### host.py
- **ChatHost class**: Orchestrates MCP client connections and LLM interactions
- Manages multiple MCP weather servers
- Implements RAG pipeline: Query → Tool Calls → Data Retrieval → LLM Response
- Interactive chat loop for user queries

### weather_Israel.py
- **MCP Server** for Israeli weather forecasts
- Uses **Playwright** to automate web browser interaction
- Tools:
  - `open_weather_forecast_israel()` - Opens browser and navigates to weather site
  - `enter_weather_forecast_city_israel()` - Enters city name for search
  - `select_weather_forecast_city_israel()` - Selects from dropdown suggestions
  - `get_weather_forecast_text()` - Gets forecast text for selected city

### weather_USA.py
- Similar structure for USA weather forecasts
- Customized for US-based weather websites

### client.py
- **MCPClient class**: Connects to MCP servers via stdio protocol
- Handles tool listing and execution
- Manages bidirectional communication with weather servers

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Chromium/Playwright browsers
- API key for Anthropic Claude (set in environment variables)
- Active internet connection for weather data retrieval

### Installation

1. **Clone or navigate to the project**:
```bash
cd c:\Users\tzvia\work\AI\mcp\project-template
```

2. **Set up virtual environment** (if not already done):
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
uv pip install -r requirements.txt
# or
pip install anthropic python-dotenv playwright mcp
```

4. **Install Playwright browsers**:
```bash
playwright install
```

5. **Set up environment variables**:
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## 🎮 Running the Project

### Start the Weather Assistant

```bash
# Make sure virtual environment is activated
python host.py
```

You'll see:
```
MCP Client Started!
Type your queries or 'quit' to exit.

Query: 
```

### Example Queries

**English queries (for USA weather)**:
```
Query: What's the weather in New York?
Query: Tell me the forecast for Los Angeles
Query: How's the weather in Chicago?
```

### How It Works

1. **User enters query** → Query is sent to Claude with available tools
2. **Claude decides which tools to use** → Makes MCP calls to weather servers
3. **Weather servers scrape data** → Uses Playwright to extract real forecast data
4. **Data is returned to Claude** → LLM sees the retrieved weather information
5. **Claude generates response** → Response is based on actual data (RAG principle)
6. **Response is displayed** → User receives weather information

## 🏗️ RAG Implementation

The project implements Retrieval-Augmented Generation:

```
Query: "What's the weather in Tel Aviv?"
  ↓
[Tool Call] open_weather_forecast_israel()
[Tool Call] enter_weather_forecast_city_israel("תל אביב")
[Tool Call] select_weather_forecast_city_israel()
[Tool Call] retrieve_weather_forecast()
  ↓
[Retrieved Data] Real weather forecast HTML/text
  ↓
Claude sees: [User Query] + [Tool Results] + [Retrieved Weather Data]
  ↓
Claude responds: "Based on the forecast, Tel Aviv will have..."
```

## 🛠️ Configuration

### System Prompt (in host.py)
Customize the weather assistant behavior by editing the `system_prompt` in `process_query()`:

```python
system_prompt = """You are a weather assistant. 
When using weather_Israel tools, 
always provide city names in Hebrew with correct spelling and right to left.
For example: תל אביב (Tel Aviv), ירושלים (Jerusalem), חיפה (Haifa)..."""
```

### Timeout Settings (in weather scripts)
- `WAIT_TIMEOUT` - Maximum wait for page load (default: 5000ms)
- `DROPDOWN_WAIT_TIME` - Wait for dropdown suggestions (default: 2000ms)

## 📦 Dependencies

- **anthropic** - Claude API client
- **playwright** - Web automation for weather scraping
- **mcp** - Model Context Protocol implementation
- **python-dotenv** - Environment variable management
- **httpx** - HTTP client for API calls

