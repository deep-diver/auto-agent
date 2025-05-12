import json
import re

def timeline_generator(text):
    """
    Generates a timeline from information extracted from text.

    Args:
        text (str): Text input related to a historical topic.

    Returns:
        str: A JSON string representing the generated timeline.

    Raises:
        ValueError: If the input text is empty or None.
    """

    if not text:
        raise ValueError("Input text cannot be empty or None.")

    # Mock data extraction - replace with actual NLP extraction in a real application
    dates = re.findall(r"\d{4}|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    events = re.split(r"(?<=[.!?])\s+", text)  # Simple sentence splitting
    figures = []
    for event in events:
        potential_figures = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", event)  # Find potential names
        figures.extend(potential_figures)
    


    timeline = []
    try:
        for i, date in enumerate(dates):
            timeline_entry = {
                "date": date,
                "event": events[i] if i < len(events) else "",
                "figures": list(set(figures)),  # Remove duplicate figures and convert to list
            }
            timeline.append(timeline_entry)
    except Exception as e:
      print(f"Error creating timeline: {e}")
      return json.dumps([])


    return json.dumps(timeline, indent=4)

