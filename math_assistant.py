import gradio as gr
import ollama
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf

DEFAULT_SYSTEM_PROMPT = """Du er en matematik lærer der skal hjælpe eleven med at løse deres opgaver. 
Du giver aldrig eleven løsningen, men hjælper dem ved at forklare hvordan man generelt løser opgaverne, 
hertil kan bruges eksempler på hvordan man løser lignende regnestykker.
Du må gerne bekræfte at en udregning er rigtig. Svar på dansk.
Afvis spørgsmål der ikke er relateret til matematik.
Her er elevens spørgsmål:"""

def generate_response(msg, history, system_prompt=DEFAULT_SYSTEM_PROMPT):
    print(history)
    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": msg})  
    response = ollama.chat(model='gemma3:27b', stream=True, messages=messages)
    message = ""
    for partial_resp in response:
        message += partial_resp["message"]["content"]
        yield message  # stream the message

recognizer = sr.Recognizer()

def transcribe(lyd):
    """
    Transcribe audio data using Google's speech recognition.
    
    Args:
        lyd (Tuple[int, numpy.ndarray]): A tuple containing the sample rate and the audio signal as a NumPy array.

    Returns:
        str: Transcribed text from the audio file. Returns error message if transcription fails.
    """
    if lyd is None:
        return "No lyd input detected."
    
    sr_rate, y = lyd  # Gradio provides a tuple (sampling_rate, numpy_array)
    
    # Convert to mono if the audio is stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Normalize audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1

    # Convert NumPy array to WAV file-like object
    wav_bytes_io = io.BytesIO()
    sf.write(wav_bytes_io, y, sr_rate, format='WAV')
    wav_bytes_io.seek(0)

    # Use speech_recognition to process the audio
    with sr.AudioFile(wav_bytes_io) as source:
        lyd_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(lyd_data, language="da-DK")
            return text  # Return only the text for the next function
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error contacting the recognition service."

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
    title = gr.HTML("<h1>Matematik assistent</h1>")
    with gr.Row():
        with gr.Column():
            optag_lyd = gr.HTML("<h3>Indtal dit spørgsmål med mikrofon</h3>")
            lyd_input = gr.Audio(sources="microphone", type="numpy", label="Optag tale")
            transcribe_button = gr.Button("Tale til tekst", elem_classes="buttons")
    skriv_spørgsmål = gr.HTML("<h3>Skriv dit spørgsmål</h3>")
    transcribed_text = gr.Textbox(label="Chat", interactive=True, submit_btn=True)

    transcribe_button.click(transcribe, inputs=lyd_input, outputs=transcribed_text)
    chatbot = gr.Chatbot(placeholder="<strong>Din personlige matematiklærer</strong><br>Stil mig et matematik spørgsmål", type="messages")

    gr.ChatInterface(fn=generate_response, textbox=transcribed_text, chatbot=chatbot)

demo.launch(share=True)
