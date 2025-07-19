"""Browser tools."""

from io import BytesIO
from time import sleep

import helium
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep

from src.driver import driver


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Save a screenshot of the current page."""
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in (
            agent.memory.steps
        ):  # Remove previous screenshots from logs for lean processing
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 2
            ):
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")  # noqa: T201
        memory_step.observations_images = [
            image.copy()  # noqa: COM812
        ]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info
        if memory_step.observations is None
        else memory_step.observations + "\n" + url_info
    )


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Search for text on the current page via Ctrl + F and jumps to the nth occurrence.

    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1).

    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        msg = f"Match n°{nth_result} not found (only {len(elements)} matches found)"
        raise Exception(msg)  # noqa: TRY002
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Close any visible modal or pop-up on the page.

    Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for
navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>
You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>
If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>
If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your
screenshot.
Never try to login in a page.
To scroll up or down, use scroll_down or scroll_up with as an argument the number of
pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by
finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>
You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
If pages seem stuck on loading, you might have to wait, for instance `import time` and
run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors =
find_all(S("ol > li"))': just look at the latest screenshot you have and read it
visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated
screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action,
it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you
can click anything else.
"""
