import json

def translate_text(text, target_language):
    """Translates text between different languages.

    Args:
        text (str): The text to be translated.
        target_language (str): The target language (e.g., "Spanish", "French", "German").

    Returns:
        str: A JSON string containing the translated text or an error message.
    """

    try:
        # Mock translation data (replace with actual translation logic if needed)
        translation_mapping = {
            "Spanish": {
                "Hello, how are you?": "Hola, ¿cómo estás?",
                "Good morning": "Buenos días",
                "Thank you": "Gracias"
            },
            "French": {
                "Hello, how are you?": "Bonjour, comment allez-vous ?",
                "Good morning": "Bonjour",
                "Thank you": "Merci"
            },
            "German": {
                "Hello, how are you?": "Hallo, wie geht es dir?",
                "Good morning": "Guten Morgen",
                "Thank you": "Danke"
            }

        }
        
        if target_language not in translation_mapping:
            raise ValueError(f"Unsupported language: {target_language}")
            
        translated_text = translation_mapping[target_language].get(text)
        if translated_text is None:
             raise ValueError(f"Translation not found for: {text}")


        result = {"translated_text": translated_text}
        return json.dumps(result)

    except ValueError as e:
        error_message = str(e)
        return json.dumps({"error": error_message})
    except Exception as e:  # Catching other potential exceptions during JSON encoding etc.
        error_message = "An unexpected error occurred during translation."
        return json.dumps({"error": error_message})
