import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Pokemon Classifier", page_icon="⚡", layout="wide")
st.title("Pokemon Classifier (Multi-Model)")
st.write("upload an image of a Pokemon and see predictions from multiple models.")

model_list = ["ResNet34", "VGG16", "GoogLeNet", "EfficientNet-B0"]
with st.sidebar:
    st.header("⚙️ Model Settings")
    selected_model = st.selectbox(
        "Select a model to use:",
        ["Compare all models (All)"] + model_list
    )
    st.info(f"Current mode: **{selected_model}**")

@st.cache_resource
def load_model_and_classes(model_name):
    classes = [
        'Abra', 'Aerodactyl', 'Alakazam', 'Alolan Sandslash', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 
        'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 
        'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 
        'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon', 'Gastly', 'Gengar', 'Geodude', 
        'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 
        'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 
        'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 
        'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'MrMime', 'Muk', 'Nidoking', 
        'Nidoqueen', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 
        'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 
        'Psyduck', 'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 
        'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 
        'Tauros', 'Tentacool', 'Tentacruel', 'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 
        'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat'
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "ResNet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        weight_path = 'resnet34.pth'
        
    elif model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes))
        weight_path = 'vgg16.pth'
        
    elif model_name == "GoogLeNet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        weight_path = 'googlenet.pth'
        
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        weight_path = 'efficientnet_b0.pth'

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        model.eval()
    except FileNotFoundError:
        st.error(f"'{weight_path}' file not found. Please train the '{model_name}' model first.")
        st.stop()
        
    return model, classes, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

upload_tab, url_tab = st.tabs(["Upload Image", "Image URL"])
image = None
with upload_tab:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

with url_tab:
    url = st.text_input("Paste Image URL here:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            st.error("Invalid URL or unable to load image.")

if image is not None:
    st.markdown("---")
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, caption='Input Image', use_container_width=True)
    
    st.markdown("---")
    
    img_tensor = transform(image).unsqueeze(0)

    if selected_model == "Compare all models (All)":
        st.subheader("Compare all models (All)")
        cols = st.columns(4)
        
        for i, m_name in enumerate(model_list):
            with cols[i]:
                st.markdown(f"### {m_name}")
                with st.spinner("Analyzing..."):
                    model, classes, device = load_model_and_classes(m_name)
                    img_tensor_device = img_tensor.to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor_device)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        top5_prob, top5_catid = torch.topk(probabilities, 5)
                        
                    for j in range(top5_prob.size(0)):
                        poke_name = classes[top5_catid[j]]
                        prob = top5_prob[j].item() * 100
                        if j == 0:
                            st.success(f"**1. {poke_name}** ({prob:.1f}%)")
                        else:
                            st.write(f"{j+1}. {poke_name} ({prob:.1f}%)")

    else:
        st.subheader(f"{selected_model} Predictions")
        with st.spinner("Analyzing..."):
            model, classes, device = load_model_and_classes(selected_model)
            img_tensor_device = img_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor_device)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                
            for i in range(top5_prob.size(0)):
                poke_name = classes[top5_catid[i]]
                prob = top5_prob[i].item() * 100
                st.write(f"**{i+1}. {poke_name}** ({prob:.2f}%)")