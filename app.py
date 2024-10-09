import sys
print("Python executable being used: ", sys.executable)
print("Python version: ", sys.version)
import streamlit as st
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image



# Model IDs
model_id_1 = "/mnt/datascience3/Boya/bouyang/llava-ftmodel—twoepoch"  # Fine-tuned LLAVA 1.6
model_id_2 = "llava-hf/llava-v1.6-vicuna-7b-hf"               # Original LLAVA 1.6

# Load models and processors
model_1 = LlavaNextForConditionalGeneration.from_pretrained(
    model_id_1, 
    torch_dtype=torch.float16,
    load_in_4bit=True
)

model_2 = LlavaNextForConditionalGeneration.from_pretrained(
    model_id_2,
    torch_dtype=torch.float16,
    load_in_4bit=True
)

processor_1 = LlavaNextProcessor.from_pretrained(model_id_1)
processor_2 = LlavaNextProcessor.from_pretrained(model_id_2)

# Streamlit app
st.title("Image to Text Generation with Model Comparison")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Generating text...")

    user_prompt = st.text_input("Enter your prompt:", "What is shown in this image?")

    prompt = f"<image>\n{user_prompt}"

    # Generate text with Fine-tuned LLAVA 1.6
    inputs_1 = processor_1(prompt, image, return_tensors="pt")
    output_1 = model_1.generate(**inputs_1, max_new_tokens=500)
    output_text_1 = processor_1.decode(output_1[0], skip_special_tokens=True)

    # Generate text with Original LLAVA 1.6
    inputs_2 = processor_2(prompt, image, return_tensors="pt")
    output_2 = model_2.generate(**inputs_2, max_new_tokens=500)
    output_text_2 = processor_2.decode(output_2[0], skip_special_tokens=True)

    # Display outputs
    st.write("**Output from Fine-tuned LLAVA 1.6:**")
    st.write(output_text_1)

    st.write("**Output from Original LLAVA 1.6:**")
    st.write(output_text_2)
