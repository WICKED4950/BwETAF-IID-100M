# **BwETAF-IID-100M**  
**Boring’s Experimental Transformer for Autoregression (Flax)** — A 100M parameter autoregressive model built in Flax. Lightweight, chaotic, and surprisingly good (I mean ok).
Because who needs sanity when you’ve got tokens to predict?

**Trained on determination, fueled by suffering, powered by free TPUs. 🔥**  

---

## 🛠️ **Model Specs**  
- **Parameters**: ~100M  
- **Context Window**: 512 tokens  
- **Dataset**: almost on 10M raw sentences (with the first 5M for a second epoch)(`WICKED4950/Raw-GPT-traindata`) Or a total of about 7.6B tokens
- **Architecture**: Custom Transformer  
- **Tokenizer**: GPT-2  
- **Trainer**: Hand-coded, Czz... Why not?
- **Final Val loss**: Almost at 3.15

---

## Why BwETAF?
- 🚀 **Built for experimentation**: Mess with the architecture guilt-free.
- ⚡ **JAX/Flax optimized**: Designed for TPU efficiency (no PyTorch bloat!).
- 🎓 **Educational focus**: Learn how transformers work under the hood.
- 💻 **Runs on potato hardware**: 100M params = no $10k GPU needed.

---

## 🚀 TPU-Optimized Training Pipeline (Proprietary)
This model was trained using a **custom JAX/Flax pipeline** optimized for free Google TPUs.  
- Trains 400M-parameter models on free TPUs (batch size ~32, ~177hrs). (In bf16)  
- Has checkpointing, saving, loading, graph plotting, Tokenization functions, Custom dataset formats for less TPU ram usage and an optimized trainer for BwETAF models 
- has a ready to use functions for anyone without touching the core part of how the model works 

Interested in the tech? Contact me for consulting/licensing.  

---

## ⚡ **Quickstart**  
Use ``` pip install BwETAF``` to install it.

** It does not include a Trainer**
```python
import BwETAF

# You can use this function for quick testing of the model
prompt = "The meaning of life is"
output = BwETAF.SetUpAPI(prompt, "WICKED4950/BwETAF-IID-100M")
print(output)  # Example: "The meaning of life is... (model's actual output)"

# Load from Hugging Face
model = BwETAF.load_hf("WICKED4950/BwETAF-IID-100M")

# Load from local directory
BwETA.load_model("path/to/model")

# Save locally
model.save_model("path/to/save")

# to get the structure and params of the model do
params = model.trainable_variables
structure = model.model_struct
```
[Open an google collab notebook](https://colab.research.google.com/drive/1v6OslzWDc1TOFwn9B2X3O_LM3J5WD4zC?usp=sharing)

---

## 🎓 Student-Friendly
As a 17-year-old solo developer, I built this to:
- Learn how LLMs work at the code level
- Experiment without corporate constraints
- Prove you don’t need $10M to train a model
Fork this repo and make it your own playground!

---

## 💬 **Important Notes**  
- This is **experimental**—expect weird bugs and cooler features.  
- It’s meant to be extended and hacked on. Go wild.  
- If it crashes, don't panic...

---

## 📩 **Reach Out**  
If you got anything to talk realted to this... Contact me at [Instagram](https://www.instagram.com/boring._.wicked)

---

## 🚧 **Upcoming Madness**  
- 🧠 **BwETAF-400M** with the same soul, but beefier body  
- 🧬 Custom layer experimentation (why not rewrite the rules?)  
- 🫠 Sanity?
