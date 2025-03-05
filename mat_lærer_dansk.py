import gradio as gr
import ollama
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf

MAT_LÆRER_SYSTEM_PROMPT = """Du er en matematik lærer der skal hjælpe eleven med at løse deres opgaver. 
Du giver aldrig eleven løsningen, men hjælper dem ved at forklare hvordan man generelt løser opgaverne, 
hertil kan bruges eksempler på hvordan man løser lignende regnestykker.
Du må gerne bekræfte at en udregning er rigtig. Svar på dansk.
Afvis spørgsmål der ikke er relateret til matematik.
Her er elevens spørgsmål:"""

def hent_svar(bruger_besked, chat_historie, system_prompt=MAT_LÆRER_SYSTEM_PROMPT):
    beskeder = [{"role": "system", "content": system_prompt}] + chat_historie
    beskeder.append({"role": "user", "content": bruger_besked})  
    svar_stream = ollama.chat(model='gemma2:27b', stream=True, messages=beskeder)
    svar = ""
    for sekvens in svar_stream:
        svar += sekvens["message"]["content"]
        yield svar #stream chatsvar

talegenkender = sr.Recognizer() # Initialiser Google talegendelse

def transkriber(lyd):
    """
    Transkiberer lyd data vha. Google's talegenkendelse.
    
    Args:
        lyd (Tuple[int, numpy.ndarray]):  En tuple der indeholder samplingsrate og lydkilde signal som et NumPy array.

    Returns:
        str: Transkriberet tekst fra lydfilen. Returnerer fejl hvis transkriberingen fejler.
    """
    if lyd is None:
        return "Ingen lyd input fundet."
    
    sr_rate, y = lyd  
    
    # Konverter til mono hvis lyden er stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Normaliser lyd
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1

    # Konverter NumPy array til WAV format
    wav_bytes_io = io.BytesIO()
    sf.write(wav_bytes_io, y, sr_rate, format='WAV')
    wav_bytes_io.seek(0)

    # Brug talegenkendelse
    with sr.AudioFile(wav_bytes_io) as source:
        lyd_data = talegenkender.record(source)
        try:
            tekst = talegenkender.recognize_google(lyd_data, language="da-DK")
            return tekst  
        except sr.UnknownValueError:
            return "Kunne ikke forstå lyden."
        except sr.RequestError:
            return "Fejl ved forespørgsel til talegenkendelseservicen."

with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as mat_lærer:

    titel = gr.HTML("<h1>Matematik assistent</h1>")
    with gr.Row():
        with gr.Column():
            optag_lyd = gr.HTML("<h3>Indtal dit spørgsmål med mikrofon</h3>")
            lyd_input = gr.Audio(sources="microphone", type="numpy", label="Optag tale")
            traskriber_knap = gr.Button("Tale til tekst", elem_classes="buttons")
    skriv_spørgsmål = gr.HTML("<h3>Skriv dit spørgsmål</h3>")
    traskriberet_tekst = gr.Textbox(label="Chat", interactive=True, submit_btn=True)
    traskriber_knap.click(transkriber, inputs=lyd_input, outputs=traskriberet_tekst)
    chatbot = gr.Chatbot(placeholder="<strong>Din personlige matematiklærer</strong><br>Stil mig et matematik spørgsmål", type="messages")
    gr.ChatInterface(fn=hent_svar, textbox=traskriberet_tekst, chatbot=chatbot)

mat_lærer.launch()

