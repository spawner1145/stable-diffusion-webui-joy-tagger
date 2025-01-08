from pathlib import Path
from argparse import ArgumentParser


def preload(parser: ArgumentParser):
    # default deepdanbooru use different paths:
    # models/deepbooru and models/torch_deepdanbooru
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc6632
    from modules import paths

    default_ddp_path = Path(paths.models_path, "JoyCaption")

    parser.add_argument(
        '--JoyCaption-path',
        type=str,
        help='Path to directory with JoyCaption project(s).',
        default=str(default_ddp_path)
    )
