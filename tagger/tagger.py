import torch
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def ensure_model_downloaded(model_name, local_base_path):
    """
    Checks if the model exists locally and downloads it if necessary.
    
    Args:
        model_name (str): The name of the model on Hugging Face.
        local_base_path (Path): Base local path where the model should be stored.
        
    Returns:
        Path: The full path to the model directory.
    """
    # Define the specific path for this model
    model_dir = local_base_path / model_name

    # Check if the model directory exists and contains all required files
    required_files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    model_files_exist = all((model_dir / file).exists() for file in required_files)

    if not model_files_exist:
        print(f"Model files do not exist. Downloading {model_name} from Hugging Face...")
        try:
            # Load processor and model from Hugging Face, saving them to the specified local path
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_dir)
            llava_model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=0, cache_dir=model_dir)

            # Save the processor and model to the local path
            processor.save_pretrained(model_dir)
            llava_model.save_pretrained(model_dir)

            print(f"Model downloaded and saved to {model_dir}")
        except Exception as e:
            print(f"Failed to download or save the model: {e}")
            return None

    else:
        print(f"Model files exist. Loading {model_name} from local directory...")

    return model_dir

def generate_caption(image):
    PROMPT = f'''Write a stable diffusion prompt for this image within 150 words. Focus on the subjects and their physical traits and physical interactions. 
    Do NOT name series or character names. 
    Do NOT use any ambiguous language. 
    Do NOT use tags or lists. 
    Do NOT describe the mood or atmosphere. 
    Do NOT describe emotions. 
    If applicable, describe sexual scenes or nudity with informal, sexual slang. 
    If a vagina (pussy), penis, butt (ass), nipples or sexual fluid (cum) is present, describe them with informal, sexual slang. 
    If characters are engaging in sex (fucking), describe the position and type of sex with informal, sexual slang. 
    If there are multiple named characters in the image, differentiate each character based on its features, and use character names instead of personal pronouns.
    '''
    # Define base path for local models directory (relative to tagger.py's location)
    base_model_path = Path("../../../../models/JoyCaption").resolve()

    # Ensure the base directory exists
    base_model_path.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

    # Ensure the model is downloaded and available locally, and get its exact path
    model_path = ensure_model_downloaded(MODEL_NAME, base_model_path)
    
    if model_path is None:
        print("Failed to load the model.")
        return None

    # Load the processor and model from the specific local path
    processor = AutoProcessor.from_pretrained(model_path)
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=0)

    # Set the model to evaluation mode
    llava_model.eval()

    with torch.no_grad():
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": PROMPT,
            },
        ]

        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
    
    return caption