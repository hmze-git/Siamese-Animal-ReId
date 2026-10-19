import gradio as gr
import torch
import os
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from SiameseModel import VGG16Siamese
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import uuid

from src.DogsData import get_val_transforms
from src.Embedding import Embedding
from src.utils import get_device

CHECKPOINT_PATH = "runs/test1/best.pt"
EMBEDDINGS_DIR = "embeddings"
TOP_K = 5
IMAGE_SIZE = 224
NEW_DOGS_DIR = "data/new_dogs"  #where we save newly enrolled photos


HMZBESTMODEL=r"D:\best.tar"

device = get_device()
Path(NEW_DOGS_DIR).mkdir(parents=True, exist_ok=True)


print("loading model and gallery...")

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model = Embedding(
    backbone=ckpt["cfg"]["backbone"],
    embedding_dim=ckpt["cfg"]["embedding_dim"],
    pretrained=False
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

#hamzah models

siameseModel=VGG16Siamese(512)
BASEDIR=Path(__file__).resolve().parent
npyDir=str(BASEDIR/'data')
npyRootName='Dogs'

siameseModel.load_state_dict(torch.load(HMZBESTMODEL,map_location=device))

siameseModel=siameseModel.to(device)
siameseModel.eval()
#HMZ EMEDDD 


emb_dir = Path(EMBEDDINGS_DIR)
gallery_embeddings = np.load(emb_dir / "embeddings.npy")

with open(emb_dir / "identities.json") as f:
    gallery_identities = json.load(f)

with open(emb_dir / "paths.json") as f:
    gallery_paths = json.load(f)

print(f"gallery loaded with {len(gallery_identities)} images")

transform = get_val_transforms(IMAGE_SIZE)


def identify_dog_ndzal(image):

    if image is None:
        return None, []

    img = np.array(image.convert("RGB"))
    img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model(img_tensor).cpu().numpy()

    similarities = cosine_similarity(query_emb, gallery_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    match_items = []
    for idx in top_indices:
        try:
            match_img = Image.open(gallery_paths[idx]).convert("RGB")
        except Exception:
            match_img = None

        score = similarities[idx]
        dog_id = gallery_identities[idx]
        caption = f"{dog_id}\nscore: {score:.3f}"
        match_items.append((match_img, caption))

    return image, match_items

def identify_dog_hmz(image):



    
    embedd=torch.from_numpy(np.load(os.path.join(npyDir,f'{npyRootName}_embedd.npy')))
    labels=np.load(os.path.join(npyDir,f'{npyRootName}_labels.npy'))
    Paths=np.load(os.path.join(npyDir,f'{npyRootName}_paths.npy'))

 

    petArr=np.array(image)

    transForms=v2.Compose([
        
                v2.Resize((224,224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32,scale=True),
                
    ])

    petTens=transForms(petArr).unsqueeze(0).to(device)
    with torch.no_grad():
        featEmbedd=siameseModel.featureEncode(petTens)
        featEmbedd=torch.nn.functional.normalize(featEmbedd,p=2,dim=1)


    embedd=embedd.to(device)


    dist=torch.cdist(featEmbedd,embedd)[0] #only need one of the vectors

    dists, idx = dist.topk(min(5, len(dist)), largest=False)

    items = []
    for i, d in zip(idx.cpu().numpy(), dists.cpu().numpy()):
            try:
                match_img = Image.open(Paths[i]).convert("RGB")
            except Exception:
                continue                      
            items.append((match_img, f"{labels[i]}\ndistance: {d:.3f}"))

    return image, items


#new dawg
def enroll_new_dog_ndzal(image, dog_name):


    global gallery_embeddings, gallery_identities, gallery_paths

    if image is None or not dog_name or not dog_name.strip():
        return "please provide both an image and a name"

    dog_name = dog_name.strip()

    #save the image so we can display it later
    save_name = f"{dog_name}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = Path(NEW_DOGS_DIR) / save_name
    image.convert("RGB").save(save_path)

    #get embedding
    img = np.array(image.convert("RGB"))
    img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        new_emb = model(img_tensor).cpu().numpy()

    #add to the in-memory gallery
    gallery_embeddings = np.vstack([gallery_embeddings, new_emb])
    gallery_identities.append(dog_name)
    gallery_paths.append(str(save_path))

    #persist everything
    np.save(emb_dir / "embeddings.npy", gallery_embeddings)
    with open(emb_dir / "identities.json", "w") as f:
        json.dump(gallery_identities, f)
    with open(emb_dir / "paths.json", "w") as f:
        json.dump(gallery_paths, f)

    embedd=np.load(os.path.join(npyDir,f'{npyRootName}_embedd.npy'))
    labels=np.load(os.path.join(npyDir,f'{npyRootName}_labels.npy'))
    Paths=np.load(os.path.join(npyDir,f'{npyRootName}_paths.npy'))
    nextId=int(labels.max())+1

        #HAMZAH LOGIC TO SAVE
    petArr=np.array(image)
    
    transForms=v2.Compose([
    
                    v2.Resize((224,224)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32,scale=True),
    
        ])
    
    
    tensImg=transForms(petArr).unsqueeze(0).to(device)
    with torch.no_grad():
            featEmbedd=siameseModel.featureEncode(tensImg)
            featEmbedd=torch.nn.functional.normalize(featEmbedd,p=2,dim=1).cpu()
            # append one row to each
    embedd = np.concatenate([embedd, featEmbedd.numpy()])
    labels = np.concatenate([labels, np.array([nextId], dtype=labels.dtype)])
    Paths  = np.concatenate([Paths, np.array([str(save_path)], dtype='U300')])
    
        # write them back
    np.save(os.path.join(npyDir,f'{npyRootName}_embedd.npy'), embedd)
    np.save(os.path.join(npyDir,f'{npyRootName}_labels.npy'), labels)
    np.save(os.path.join(npyDir,f'{npyRootName}_paths.npy'), Paths)
    



    return f"successfully added **{dog_name}** to the gallery.\n\ntotal images now: {len(gallery_identities)}"

def enroll_new_dog_hmz(image,dog_name):

    if image is None or not dog_name or not dog_name.strip():
        return "please provide both an image and a name"


    if image is None or not dog_name or not dog_name.strip():
        return "please provide both an image and a name"
    
    dog_name = dog_name.strip()
    
    #save the image so we can display it later
    save_name = f"{dog_name}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = Path(NEW_DOGS_DIR) / save_name
    image.convert("RGB").save(save_path)


    
    return f"successfully added **{dog_name}** to the gallery."

#GRADIO INTERFAC

#gr.tab gives you new tabs
#put your code on the siemese tab

with gr.Blocks(title="Find my Dawg") as demo:
    gr.Markdown("# Dog Re-Identification")
    gr.Markdown("upload a photo to find matching dogs, or enrol a completely new dog into the system.")

    with gr.Tab("Identify Dog with Distance"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="upload dog photo")
                search_btn = gr.Button("find matches", variant="primary")
            with gr.Column(scale=1):
                query_preview = gr.Image(label="your upload", interactive=False)

        gr.Markdown("### top matches")
        match_gallery = gr.Gallery(
            label="top 5 matches",
            columns=5,
            rows=1,
            height=240,
            object_fit="contain"
        )

        search_btn.click(
            fn=identify_dog_ndzal,
            inputs=input_image,
            outputs=[query_preview, match_gallery]
        )

    with gr.Tab("Identify Dog with Siemese"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="upload dog photo")
                search_btn = gr.Button("find matches", variant="primary")
            with gr.Column(scale=1):
                query_preview = gr.Image(label="your upload", interactive=False)

        gr.Markdown("### top matches")
        match_gallery = gr.Gallery(
            label="top 5 matches",
            columns=5,
            rows=1,
            height=240,
            object_fit="contain"
        )

        search_btn.click(
            fn=identify_dog_hmz,
            inputs=input_image,
            outputs=[query_preview, match_gallery]
        )


    with gr.Tab("Enroll New Dog"):
        gr.Markdown(
            "use this when the dog is **not** already in the gallery. the system will add it immediately (no retraining needed).")
        with gr.Row():
            with gr.Column():
                enroll_image = gr.Image(type="pil", label="photo of the new dog")
                dog_name = gr.Textbox(label="name / id for this dog", placeholder="e.g. Buddy_01")
                enroll_btn = gr.Button("add to gallery", variant="primary")
            with gr.Column():
                enroll_status = gr.Markdown()

        enroll_btn.click(
            fn=enroll_new_dog_ndzal,
            inputs=[enroll_image, dog_name],
            outputs=enroll_status
        )

if __name__ == "__main__":
    demo.launch()