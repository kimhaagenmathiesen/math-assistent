import gradio as gr
import ollama
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf
""" 
Virker fint, giv den evt. adgang til tools i form af numpy, math osv.
"""


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
    response = ollama.chat(model='gemma2:27b', stream=True, messages=messages)
    message = ""
    for partial_resp in response:
        message += partial_resp["message"]["content"]
        yield message #stream beskeden

recognizer = sr.Recognizer()

def transcribe(lyd):
    """
    Transkibere lyd data vha. Google's talegenkendelse.
    
    Args:
        lyd (Tuple[int, numpy.ndarray]):  Et tuple der indeholder samplingsrate og lydkilende signal som en NumPy array.

    Returns:
        str: Transkriberet tekst fra lydfilen. Returnerer fejl hvis transkriberingen fejler.
    """
    if lyd is None:
        return "No lyd input detected."
    
    sr_rate, y = lyd  # Gradio provides a tuple (sampling_rate, numpy_array)
    
    # Konverter til mono hvis lyden er stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Normaliser lyd
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1

    # Konverter NumPy array til WAV file-like object
    wav_bytes_io = io.BytesIO()
    sf.write(wav_bytes_io, y, sr_rate, format='WAV')
    wav_bytes_io.seek(0)

    # Use speech_recognition to process the lyd
    with sr.AudioFile(wav_bytes_io) as source:
        lyd_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(lyd_data, language="da-DK")
            return text  # Return only the text for the next function
        except sr.UnknownValueError:
            return "Kunne ikke forstå lyden."
        except sr.RequestError:
            return "Fejl ved forespørgsel til genkendelsesservice."

"""
chatbot = gr.ChatInterface(
    fn=generate_response,
    chatbot=gr.Chatbot(avatar_images=["user.jpg", "chatbot.png"], height="50vh", type="messages"),  
    additional_inputs=[gr.Textbox(DEFAULT_SYSTEM_PROMPT, label="System Prompt")],
    
    title="Gemma-2 (27B) Chatbot using Ollama",
    description="Feel free to ask any question.",
    theme="soft",
    submit_btn="⬅ Send"
)

chatbot.launch()
"""
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

