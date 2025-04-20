import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pathlib import Path
import os
from PIL import Image
import glob

# 尝试导入 ModelScope
try:
    from modelscope import snapshot_download as ms_snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

# 定义模型路径和仓库
MODEL_DIR = Path("models/joy_caption")
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
MODELSCOPE_MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"  # 根据实际 ModelScope 仓库名调整

# CAPTION_TYPE_MAP 定义多种 caption 类型
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tag list for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# 全局变量，用于存储模型和处理器
processor = None
llava_model = None

# 延迟加载模型的函数
def load_models(
    download_source: str = "huggingface",
    torch_dtype: str = "bfloat16",

    device_map: str = "cuda:0",
    patch_size: int = 14,
    vision_feature_select_strategy: str = "default"
):
    global processor, llava_model

    # 验证下载源
    if download_source not in ["huggingface", "modelscope"]:
        raise ValueError("download_source must be 'huggingface' or 'modelscope'")

    if download_source == "modelscope" and not MODELSCOPE_AVAILABLE:
        raise ImportError("ModelScope is not installed. Please install modelscope or use 'huggingface' as the download source.")

    # 转换 torch_dtype 字符串为 torch.dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}. Choose from {list(dtype_map.keys())}")
    torch_dtype = dtype_map[torch_dtype]

    # 验证 device_map
    valid_device_maps = ["cpu", "cuda:0", "auto", "balanced", "balanced_low_0", "sequential"]
    if device_map not in valid_device_maps:
        raise ValueError(f"Invalid device_map: {device_map}. Choose from {valid_device_maps}")

    # 确保模型目录存在
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 检查模型文件是否存在（使用 config.json 作为标志文件）
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        print(f"Downloading JoyCaption model to {MODEL_DIR} from {download_source}")
        if download_source == "huggingface":
            snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False, token=HF_TOKEN)
        elif download_source == "modelscope":
            ms_snapshot_download(model_id=MODELSCOPE_MODEL_NAME, cache_dir=MODEL_DIR)

    # 加载模型
    print("Loading JoyCaption")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    # 设置 patch_size 和 vision_feature_select_strategy 以避免警告
    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True
    )
    llava_model.eval()

# 卸载模型的函数
def unload_models():
    global processor, llava_model
    try:
        if llava_model is not None:
            # 释放模型显存
            llava_model.to('cpu')
            del llava_model
            llava_model = None
        if processor is not None:
            del processor
            processor = None
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Model and processor successfully unloaded from memory."
    except Exception as e:
        return f"Error unloading model: {str(e)}"

# 推理函数（单张图片）
@torch.no_grad()
def stream_chat(
    input_image: Image.Image,
    caption_type: str,
    caption_length: str | int,
    extra_options: list[str],
    name_input: str,
    custom_prompt: str,
    download_source: str = "huggingface",
    torch_dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    patch_size: int = 14,
    vision_feature_select_strategy: str = "default",
    max_new_tokens: int = 300,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_k: int = 0,
    top_p: float = 0.9
) -> tuple[str, str]:
    # 在推理时加载模型（仅第一次调用时加载）
    if 'llava_model' not in globals() or llava_model is None:
        load_models(
            download_source=download_source,
            torch_dtype=torch_dtype,
            device_map=device_map,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy
        )

    torch.cuda.empty_cache()

    # 处理 top_k（UI 传入 0 表示禁用 top_k）
    top_k = None if top_k == 0 else top_k

    # 处理 caption 长度
    length = None if caption_length == "any" else caption_length
    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass
    
    # 构建 prompt
    map_idx = 0 if length is None else (1 if isinstance(length, int) else 2)
    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # 添加额外选项
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)
    
    # 替换 name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    # 使用自定义 prompt（如果提供）
    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    
    print(f"Prompt: {prompt_str}")

    # 构建对话
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt_str},
    ]

    # 格式化对话
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    # 处理输入
    inputs = processor(text=[convo_string], images=[input_image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)  # 保持与参考脚本一致

    # 生成 caption
    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        suppress_tokens=None,
        use_cache=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )[0]

    # 裁剪 prompt 部分
    generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

    # 解码 caption
    caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption = caption.strip()

    return prompt_str, caption

# 批量推理函数
def batch_caption(
    input_directory: str,
    caption_type: str,
    caption_length: str | int,
    extra_options: list[str],
    name_input: str,
    custom_prompt: str,
    download_source: str = "huggingface",
    torch_dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    patch_size: int = 14,
    vision_feature_select_strategy: str = "default",
    max_new_tokens: int = 300,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_k: int = 0,
    top_p: float = 0.9
) -> str:
    """
    Process all images in the specified directory and save captions to .txt files with the same name.
    Returns a status message summarizing the process.
    """
    try:
        # 验证目录是否存在
        input_dir = Path(input_directory)
        if not input_dir.exists() or not input_dir.is_dir():
            return f"Error: Directory '{input_directory}' does not exist or is not a directory."

        # 支持的图片扩展名
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        image_files = [f for f in input_dir.glob("*") if f.suffix.lower() in image_extensions]

        if not image_files:
            return f"Error: No images found in directory '{input_directory}'."

        status_messages = [f"Found {len(image_files)} images in '{input_directory}'.\nProcessing images...\n"]
        
        for image_path in image_files:
            try:
                # 加载图片
                image = Image.open(image_path).convert("RGB")
                status_messages.append(f"Processing '{image_path.name}'...")

                # 生成 caption
                prompt, caption = stream_chat(
                    input_image=image,
                    caption_type=caption_type,
                    caption_length=caption_length,
                    extra_options=extra_options,
                    name_input=name_input,
                    custom_prompt=custom_prompt,
                    download_source=download_source,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    patch_size=patch_size,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # 保存 caption 到同名 .txt 文件
                txt_path = image_path.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                
                status_messages.append(f"  Successfully generated caption for '{image_path.name}' and saved to '{txt_path.name}'.")
            
            except Exception as e:
                status_messages.append(f"  Error processing '{image_path.name}': {str(e)}")
        
        status_messages.append("\nBatch processing completed.")
        return "\n".join(status_messages)
    
    except Exception as e:
        return f"Error during batch processing: {str(e)}"