import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import io

#  Google API Key setup
genai.configure(api_key="AIzaSyDCToalcS0jGdZyNiFxRnJOnDkoWCYd6z")

#  Streamlit layout
st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")
st.title("🧠 AI Medical Diagnosis with Image + Voice")
st.markdown("Upload your **medical image** and describe your **symptoms by voice** for AI-based diagnosis.")
#streamlit interface

#  Upload image
uploaded_image = st.file_uploader("Upload Medical Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

#  Voice input
record_voice = st.button("🎤 Record Symptoms")
#voice input for taking voice record
voice_text = ""
if record_voice:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Recording... Speak now.")
        audio = recognizer.listen(source, timeout=5)
    try:
        voice_text = recognizer.recognize_google(audio)
        st.success(f"🗣️ Recognized: {voice_text}")
    except sr.UnknownValueError:
        st.error("Could not understand your voice.")
    except sr.RequestError:
        st.error("Speech recognition service failed.")

# Generate diagnosis
if st.button("🧪 Run Diagnosis") and uploaded_image:
    image_data = uploaded_image.read()

    # Image + voice fusion prompt
    prompt_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image_data
        },
        f"""The patient reports the following symptoms: "{voice_text}". 
        Analyze the uploaded medical image along with the described symptoms.
        Provide:
        - Detailed image interpretation
        - Disease diagnosis
        - Recommendations (tests, treatment)
        - Confidence level
        - Ethical consideration if any
        Use clear and clinical tone."""
    ]

    # Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_parts)

    st.subheader("🩺 AI Diagnosis")
    st.write(response.text)

