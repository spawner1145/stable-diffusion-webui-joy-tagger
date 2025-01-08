import modules.scripts as scripts
import gradio as gr
import os
from PIL import Image
from .tagger import generate_caption

from modules import script_callbacks

def model_start(image):
    if image is not None:
        # 调用图生文模型进行推理
        caption = generate_caption(image)
        return caption
    else:
        return "请先上传一张图片。"

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css="""
        .custom-textbox {
            height: 200px;
            overflow-y: auto;
            scrollbar-width: thin; /* Firefox */
            -ms-overflow-style: none;  /* Internet Explorer 10+ */
        }
        .custom-textbox::-webkit-scrollbar { 
            width: 8px; /* 滚动条宽度 */
        }
        .custom-textbox::-webkit-scrollbar-thumb {
            background-color: #888; /* 滚动条颜色 */
            border-radius: 4px;
        }
        .custom-textbox::-webkit-scrollbar-thumb:hover {
            background-color: #555; /* 鼠标悬停时滚动条颜色 */
        }
    """) as tagger_interface:
        with gr.Row():
            image_input = gr.Image(label="upload an image", source="upload")
            text_output = gr.Textbox(
                label="outputs",
                elem_classes="custom-textbox"  # 添加自定义CSS类
            )
        
        with gr.Row():
            run_tagger = gr.Button("run JoyTagger")
            t2i = gr.Button("send to txt2img")
            i2i = gr.Button("send to img2img")

        run_tagger.click(fn=model_start, outputs=text_output)
        t2i.click(fn=button2_click, outputs=text_output)
        i2i.click(fn=button3_click, outputs=text_output)
        image_input.change(fn=model_start, inputs=image_input, outputs=text_output)
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [(tagger_interface, "JoyTagger", "joytagger_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)