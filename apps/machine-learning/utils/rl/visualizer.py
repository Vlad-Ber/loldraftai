import os
from io import BytesIO
import requests
from PIL import Image
import numpy as np
from utils import DATA_DIR


class LoLDraftVisualizer:
    def __init__(self):
        self.cache_dir = os.path.join(DATA_DIR, "champion_icons")
        self.icon_size = (120, 120)
        self.default_icon_id = -1
        self.base_url = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/"

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_champion_icon(self, champion_id):
        icon_path = os.path.join(self.cache_dir, f"{champion_id}.png")

        if not os.path.exists(icon_path):
            url = f"{self.base_url}{champion_id}.png"
            response = requests.get(url)

            if response.status_code == 200:
                with open(icon_path, "wb") as f:
                    f.write(response.content)
            else:
                icon_path = os.path.join(self.cache_dir, f"{self.default_icon_id}.png")
                if not os.path.exists(icon_path):
                    default_url = f"{self.base_url}{self.default_icon_id}.png"
                    default_response = requests.get(default_url)
                    with open(icon_path, "wb") as f:
                        f.write(default_response.content)

        return Image.open(icon_path).resize(self.icon_size)

    def create_draft_image(self, blue_team, red_team):
        draft_image = Image.new("RGB", (self.icon_size[0] * 2, self.icon_size[1] * 5))

        for i, (blue_champ, red_champ) in enumerate(zip(blue_team, red_team)):
            blue_icon = self.get_champion_icon(blue_champ)
            red_icon = self.get_champion_icon(red_champ)

            draft_image.paste(blue_icon, (0, i * self.icon_size[1]))
            draft_image.paste(red_icon, (self.icon_size[0], i * self.icon_size[1]))

        return draft_image

    def save_draft_image(self, blue_team, red_team, output_path):
        draft_image = self.create_draft_image(blue_team, red_team)
        draft_image.save(output_path)

    def get_draft_array(self, blue_team, red_team):
        draft_image = self.create_draft_image(blue_team, red_team)
        return np.array(draft_image)


def integrate_with_env(env):
    class LoLDraftEnvWithRender(env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.visualizer = LoLDraftVisualizer()
            self.draft_image = None


        def _is_draft_complete(self):
            is_complete = super()._is_draft_complete()
            if is_complete:
                blue_team = np.argmax(self.blue_ordered_picks, axis=1)
                red_team = np.argmax(self.red_ordered_picks, axis=1)
                self.draft_image = self.visualizer.get_draft_array(blue_team, red_team)
            return is_complete
              

        def render(self):
            # this could return draft_image of last draft, but it helps also getting an image in vectorized envs
            # because vectorized envs automatically reset so it's hard to call render when state is done
            if self.draft_image is not None:
                # Convert numpy array to PIL Image
                img = Image.fromarray(self.draft_image.astype('uint8'), 'RGB')
                # Create a bytes buffer
                buffer = BytesIO()
                # Save the image to the buffer in PNG format
                img.save(buffer, format="PNG")
                # Get the bytes data
                image_data = buffer.getvalue()
                return image_data
            return None

    return LoLDraftEnvWithRender
