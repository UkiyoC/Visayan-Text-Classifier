import re
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import customtkinter as ctk
from PIL import Image

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
classification_model = tf.keras.models.load_model('classification_model/')

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['Cebuano', 'Hiligaynon', 'Waray']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

def detect_language(text):
    if not re.search(r'[a-zA-Z]', text):
        return "Error: Input text contains only numbers or special characters"
    
    processed_data = prepare_data(text, tokenizer)
    prediction = make_prediction(classification_model, processed_data)
    return prediction

def highlight_language():
    input_text = text_entry.get("1.0", "end-1c")
    detected_language = detect_language(input_text)
    cebuano_label.configure(text_color="white", fg_color="gray10")
    hiligaynon_label.configure(text_color="white", fg_color="gray10")
    waray_label.configure(text_color="white", fg_color="gray10")
    error_label.configure(text="")
    if detected_language == "Cebuano":
        cebuano_label.configure(fg_color="#fca5a5")
    elif detected_language == "Hiligaynon":
        hiligaynon_label.configure(fg_color="#86efac")
    elif detected_language == "Waray":
        waray_label.configure(fg_color="#7ed3fc")
    elif detected_language.startswith("Error:"):
        error_label.configure(text=detected_language, text_color="red", wraplength=700)

def reset_gui():
    text_entry.delete("1.0", "end")
    cebuano_label.configure(text_color="white", fg_color="gray10")
    hiligaynon_label.configure(text_color="white", fg_color="gray10")
    waray_label.configure(text_color="white", fg_color="gray10")
    error_label.configure(text="")

app = ctk.CTk()
app.title("Visayan Text Classifier")
app.geometry("900x600")

# Set the window to be centered
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_coordinate = (screen_width - 900) // 2
y_coordinate = (screen_height - 600) // 2
app.geometry(f"900x600+{x_coordinate}+{y_coordinate}")

# Add padding to the window
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(2, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(3, weight=1)
app.grid_columnconfigure(2, weight=1)

# Fonts
text_font = ctk.CTkFont("Verdana", 16)
label_font = ctk.CTkFont("Verdana", 18)

# GUI Elements
enter_text_label = ctk.CTkLabel(app, text="Enter any Visayan text:", font=text_font)
enter_text_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

text_entry = ctk.CTkTextbox(app, height=300, width=850, font=text_font)
text_entry.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")

error_label = ctk.CTkLabel(app, text="", font=text_font, text_color="red", wraplength=600)
error_label.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="w")

button_frame = ctk.CTkFrame(app)
button_frame.grid(row=2, column=2, padx=10, pady=10, sticky="e")

enter_button = ctk.CTkButton(
    button_frame,
    text="Enter",
    command=highlight_language,
    font=label_font,
    width=150,
    height=50,
    fg_color="#2663eb",
    hover_color="#1a4be3",
    text_color="white"
)
enter_button.pack(side="right", padx=5)

reset_image = ctk.CTkImage(Image.open("img/reset.png"), size=(20, 20))
reset_button = ctk.CTkButton(
    button_frame,
    image=reset_image,
    text="",
    command=reset_gui,
    font=label_font,
    width=50,
    height=50,
    fg_color="#991b1b",
    hover_color="#cc3333",
)
reset_button.pack(side="right", padx=5)

cebuano_label = ctk.CTkLabel(app, text="Cebuano", text_color="white", fg_color="gray10", corner_radius=8, font=label_font)
hiligaynon_label = ctk.CTkLabel(app, text="Hiligaynon", text_color="white", fg_color="gray10", corner_radius=8, font=label_font)
waray_label = ctk.CTkLabel(app, text="Waray-Waray", text_color="white", fg_color="gray10", corner_radius=8, font=label_font)

label_width = 240
cebuano_label.configure(width=label_width)
hiligaynon_label.configure(width=label_width)
waray_label.configure(width=label_width)

cebuano_label.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
hiligaynon_label.grid(row=3, column=1, padx=20, pady=20, sticky="nsew")
waray_label.grid(row=3, column=2, padx=20, pady=20, sticky="nsew")

app.mainloop()