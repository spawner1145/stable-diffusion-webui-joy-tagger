try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
import gradio as gr
from .inference import stream_chat, unload_models, batch_caption

# 定义标题
TITLE = "<h1><center>JoyCaption</center></h1>"

# 构建 Gradio 界面
def create_ui():
    with gr.Blocks() as demo:
        gr.HTML(TITLE)

        with gr.Tabs():
            # Single Image Tab
            with gr.Tab("Single Image Caption"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="Input Image")
                        caption_type = gr.Dropdown(
                            choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                            label="Caption Type",
                            value="Descriptive",
                        )
                        caption_length = gr.Dropdown(
                            choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                    [str(i) for i in range(20, 261, 10)],
                            label="Caption Length",
                            value="long",
                        )
                        extra_options = gr.CheckboxGroup(
                            choices=[
                                "If there is a person/character in the image you must refer to them as {name}.",
                                "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                                "Include information about lighting.",
                                "Include information about camera angle.",
                                "Include information about whether there is a watermark or not.",
                                "Include information about whether there are JPEG artifacts or not.",
                                "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                                "Do NOT include anything sexual; keep it PG.",
                                "Do NOT mention the image's resolution.",
                                "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                                "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                                "Do NOT mention any text that is in the image.",
                                "Specify the depth of field and whether the background is in focus or blurred.",
                                "If applicable, mention the likely use of artificial or natural lighting sources.",
                                "Do NOT use any ambiguous language.",
                                "Include whether the image is sfw, suggestive, or nsfw.",
                                "ONLY describe the most important elements of the image."
                            ],
                            label="Extra Options"
                        )
                        name_input = gr.Textbox(label="Person/Character Name (if applicable)")
                        gr.Markdown("**Note:** Name input is only used if an Extra Option is selected that requires it.")
                        custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")
                        gr.Markdown("**Note:** Alpha Two is not a general instruction follower and will not follow prompts outside its training data well. Use this feature with caution.")
                        download_source = gr.Dropdown(
                            choices=["huggingface", "modelscope"],
                            label="Model Download Source",
                            value="huggingface",
                        )

                        # Advanced settings for model and generation parameters
                        with gr.Accordion("Advanced Model and Generation Settings", open=False):
                            torch_dtype = gr.Dropdown(
                                choices=["bfloat16", "float16", "float32"],
                                label="Torch Data Type",
                                value="bfloat16",
                            )
                            device_map = gr.Dropdown(
                                choices=["cuda:0", "cpu", "auto", "balanced", "balanced_low_0", "sequential"],
                                label="Device Map",
                                value="cuda:0",
                            )
                            patch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                step=1,
                                label="Patch Size",
                                value=14,
                            )
                            vision_feature_select_strategy = gr.Dropdown(
                                choices=["default", "full"],
                                label="Vision Feature Select Strategy",
                                value="default",
                            )
                            max_new_tokens = gr.Slider(
                                minimum=50,
                                maximum=1000,
                                step=10,
                                label="Max New Tokens",
                                value=300,
                            )
                            do_sample = gr.Checkbox(
                                label="Enable Sampling",
                                value=True,
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Temperature",
                                value=0.6,
                            )
                            top_k = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                label="Top K (set to 0 to disable)",
                                value=0,
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                label="Top P",
                                value=0.9,
                            )

                        run_button = gr.Button("Caption")
                        unload_button = gr.Button("Unload Model")
                    
                    with gr.Column():
                        output_prompt = gr.Textbox(label="Prompt that was used")
                        output_caption = gr.Textbox(label="Caption")
                        unload_status = gr.Textbox(label="Unload Status")
                
                # Run button click event for single image
                run_button.click(
                    fn=stream_chat,
                    inputs=[
                        input_image,
                        caption_type,
                        caption_length,
                        extra_options,
                        name_input,
                        custom_prompt,
                        download_source,
                        torch_dtype,
                        device_map,
                        patch_size,
                        vision_feature_select_strategy,
                        max_new_tokens,
                        do_sample,
                        temperature,
                        top_k,
                        top_p,
                    ],
                    outputs=[output_prompt, output_caption]
                )
                unload_button.click(
                    fn=unload_models,
                    inputs=[],
                    outputs=unload_status
                )

            # Batch Caption Tab
            with gr.Tab("Batch Caption"):
                with gr.Row():
                    with gr.Column():
                        input_directory = gr.Textbox(label="Input Directory Path", placeholder="Enter the path to the directory containing images")
                        batch_caption_type = gr.Dropdown(
                            choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                            label="Caption Type",
                            value="Descriptive",
                        )
                        batch_caption_length = gr.Dropdown(
                            choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                    [str(i) for i in range(20, 261, 10)],
                            label="Caption Length",
                            value="long",
                        )
                        batch_extra_options = gr.CheckboxGroup(
                            choices=[
                                "If there is a person/character in the image you must refer to them as {name}.",
                                "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                                "Include information about lighting.",
                                "Include information about camera angle.",
                                "Include information about whether there is a watermark or not.",
                                "Include information about whether there are JPEG artifacts or not.",
                                "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                                "Do NOT include anything sexual; keep it PG.",
                                "Do NOT mention the image's resolution.",
                                "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                                "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                                "Do NOT mention any text that is in the image.",
                                "Specify the depth of field and whether the background is in focus or blurred.",
                                "If applicable, mention the likely use of artificial or natural lighting sources.",
                                "Do NOT use any ambiguous language.",
                                "Include whether the image is sfw, suggestive, or nsfw.",
                                "ONLY describe the most important elements of the image."
                            ],
                            label="Extra Options"
                        )
                        batch_name_input = gr.Textbox(label="Person/Character Name (if applicable)")
                        gr.Markdown("**Note:** Name input is only used if an Extra Option is selected that requires it.")
                        batch_custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")
                        gr.Markdown("**Note:** Alpha Two is not a general instruction follower and will not follow prompts outside its training data well. Use this feature with caution.")
                        batch_download_source = gr.Dropdown(
                            choices=["huggingface", "modelscope"],
                            label="Model Download Source",
                            value="huggingface",
                        )

                        # Advanced settings for model and generation parameters
                        with gr.Accordion("Advanced Model and Generation Settings", open=False):
                            batch_torch_dtype = gr.Dropdown(
                                choices=["bfloat16", "float16", "float32"],
                                label="Torch Data Type",
                                value="bfloat16",
                            )
                            batch_device_map = gr.Dropdown(
                                choices=["cuda:0", "cpu", "auto", "balanced", "balanced_low_0", "sequential"],
                                label="Device Map",
                                value="cuda:0",
                            )
                            batch_patch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                step=1,
                                label="Patch Size",
                                value=14,
                            )
                            batch_vision_feature_select_strategy = gr.Dropdown(
                                choices=["default", "full"],
                                label="Vision Feature Select Strategy",
                                value="default",
                            )
                            batch_max_new_tokens = gr.Slider(
                                minimum=50,
                                maximum=1000,
                                step=10,
                                label="Max New Tokens",
                                value=300,
                            )
                            batch_do_sample = gr.Checkbox(
                                label="Enable Sampling",
                                value=True,
                            )
                            batch_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Temperature",
                                value=0.6,
                            )
                            batch_top_k = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                label="Top K (set to 0 to disable)",
                                value=0,
                            )
                            batch_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                label="Top P",
                                value=0.9,
                            )

                        batch_run_button = gr.Button("Batch Caption")
                        batch_unload_button = gr.Button("Unload Model")
                    
                    with gr.Column():
                        batch_output_status = gr.Textbox(label="Batch Processing Status", lines=10)
                        batch_unload_status = gr.Textbox(label="Unload Status")
                
                # Run button click event for batch captioning
                batch_run_button.click(
                    fn=batch_caption,
                    inputs=[
                        input_directory,
                        batch_caption_type,
                        batch_caption_length,
                        batch_extra_options,
                        batch_name_input,
                        batch_custom_prompt,
                        batch_download_source,
                        batch_torch_dtype,
                        batch_device_map,
                        batch_patch_size,
                        batch_vision_feature_select_strategy,
                        batch_max_new_tokens,
                        batch_do_sample,
                        batch_temperature,
                        batch_top_k,
                        batch_top_p,
                    ],
                    outputs=[batch_output_status]
                )
                batch_unload_button.click(
                    fn=unload_models,
                    inputs=[],
                    outputs=batch_unload_status
                )
    return demo

if IN_WEBUI:
    def on_ui_tabs():
        block = create_ui()
        return [(block, "JoyCaption", "JoyCaption_tab")]
    script_callbacks.on_ui_tabs(on_ui_tabs)
else:
    block = create_ui()
    block.launch()