import asyncio
from playwright.async_api import async_playwright, Page, Browser
from mcp.server.fastmcp import FastMCP
from playwright.async_api import TimeoutError as PlaywrightTimeout

mcp = FastMCP("weather-Israel")

FORECAST_URL = "https://www.weather2day.co.il/forecast"
WAIT_TIMEOUT = 5000   # milliseconds
DROPDOWN_WAIT_TIME = 2000  # milliseconds

# Global state to persist browser and page across tool calls
_browser: Browser | None = None
_page: Page | None = None
_playwright = None


@mcp.tool()
async def open_weather_forecast_israel() -> dict:
    """
    Tool 1: פותח את הדפדפן ומנווט לדף מזג האוויר הישראלי.
    Opens the browser and navigates to the Israeli weather forecast page.
    """
    global _browser, _page, _playwright

    # Close any existing session cleanly
    if _browser:
        await _browser.close()
        _browser = None
        _page = None
    if _playwright:
        await _playwright.stop()
        _playwright = None

    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=False)
    _page = await _browser.new_page()

    await _page.goto(FORECAST_URL, wait_until="domcontentloaded")
    await _page.wait_for_load_state("networkidle")
    # Extra wait for JS frameworks (React/Angular/Vue) to fully render
    await _page.wait_for_timeout(2000)

    return {
        "status": "success",
        "message": f"הדפדפן נפתח והדף נטען: {FORECAST_URL}",
        "url": _page.url,
    }


@mcp.tool()
async def enter_weather_forecast_city_israel(city_name: str) -> dict:
    global _page

    if _page is None:
        raise RuntimeError(
            "Browser not open. Call open_weather_forecast_israel first."
        )

    if isinstance(city_name, bytes):
        city_name = city_name.decode("utf-8")

    # Fill the input field with id = city_search_forecast
    await _page.locator("#city_search_forecast").fill(city_name)

    await _page.wait_for_timeout(DROPDOWN_WAIT_TIME)
    return {
        "status": "success",
        "message": f"City '{city_name}' entered.",
        "city": city_name,
    }

@mcp.tool()
async def select_weather_forecast_city_israel() -> dict:
    """
    Tool 3: בוחר את הפריט הראשון ברשימת הערים המוצעת.
    Selects the first item from the city suggestions dropdown list.
    """
    global _page

    if _page is None:
        raise RuntimeError(
            "Browser not open. Call open_weather_forecast_israel first."
        )

    # Locator for the autocomplete dropdown list first item
    first_option = _page.locator('#city_search_forecastautocomplete-list div').first

    # Get the text content of the first option
    city_text = await first_option.text_content()
    city_text_stripped = city_text.strip() if city_text else "Unknown"

    # Click the first option
    await first_option.click()

    try:
        await _page.wait_for_load_state("networkidle", timeout=WAIT_TIMEOUT)
    except PlaywrightTimeout:
        pass

    return {
        "status": "success",
        "message": f"העיר '{city_text_stripped}' נבחרה בהצלחה.",
        "selected_city": city_text_stripped,
        "current_url": _page.url,
    }


@mcp.tool()
async def close_browser() -> dict:
    """
    Helper: closes the browser and releases resources.
    """
    global _browser, _page, _playwright

    if _browser:
        await _browser.close()
        _browser = None
        _page = None

    if _playwright:
        await _playwright.stop()
        _playwright = None

    return {"status": "success", "message": "הדפדפן נסגר בהצלחה."}

@mcp.tool()
async def get_weather_forecast_text() -> dict:
    """
    Tool 4: 
    Retrieves the weather forecast for the selected city in text format.
    """
    global _page

    if _page is None:
        raise RuntimeError(
            "Browser not open. Call open_weather_forecast_israel first."
        )
    #forecast_text = await _page.inner_text("div#ecmwf.forecast-source")
    
    ecmwf_div = _page.locator('div#ecmwf.forecast-source')
    text_content = await ecmwf_div.inner_text()  
    
    if not text_content:
        return {
            "status": "error",
            "message": "ECMWF div found but contains no text content.",
            "content": ""
        }
    
    return {
        "status": "success",
        "forecast": text_content.strip(),
    }

# ── Demo / manual test ────────────────────────────────────────────────────────
async def main():
    print("=== MCP Weather Tools Demo ===\n")

    print("1. פותח דפדפן ומנווט לאתר מזג האוויר...")
    result = await open_weather_forecast_israel()
    print(f"   {result}\n")

    print("2. מזין שם עיר: 'תל אביב'...")
    result = await enter_weather_forecast_city_israel("תל אביב")
    print(f"   {result}\n")

    print("3. בוחר עיר ראשונה מהרשימה...")
    result = await select_weather_forecast_city_israel()
    print(f"   {result}\n")
    result = await get_weather_forecast_text()
    print(f"   {result}\n")

if __name__ == "__main__":
    mcp.run(transport="stdio")
        #asyncio.run(main())

