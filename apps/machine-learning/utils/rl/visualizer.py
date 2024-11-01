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

        def render(self):
            """Render the current state of the draft"""
            # Only render if we have picks
            if (
                np.sum(self.blue_ordered_picks) > 0
                or np.sum(self.red_ordered_picks) > 0
            ):
                blue_team = np.argmax(self.blue_ordered_picks, axis=1)
                red_team = np.argmax(self.red_ordered_picks, axis=1)

                # Create the draft image
                draft_image = self.visualizer.get_draft_array(blue_team, red_team)

                # Convert numpy array to PNG bytes
                img = Image.fromarray(draft_image.astype("uint8"), "RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()
            return None

    return LoLDraftEnvWithRender
