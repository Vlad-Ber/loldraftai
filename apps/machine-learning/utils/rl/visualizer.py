import os
from io import BytesIO
import requests
from PIL import Image
import numpy as np
from utils import DATA_DIR
from utils.rl.env import FlexibleRoleDraftEnv


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
    env: FlexibleRoleDraftEnv

    class LoLDraftEnvWithRender(env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.visualizer = LoLDraftVisualizer()

        def render(self):
            """Render the current state of the draft"""
            # Only render if we have picks and roles assigned
            if len(self.state.blue_picks) > 0 or len(self.state.red_picks) > 0:
                # Get ordered picks based on role assignments
                blue_team = np.zeros(5, dtype=np.int32)
                red_team = np.zeros(5, dtype=np.int32)

                # Fill in assigned picks in role order
                for role_idx, role in enumerate(self.roles):
                    # Find blue champion for this role
                    for champ_id, assigned_role in self.state.blue_roles.items():
                        if assigned_role == role:
                            blue_team[role_idx] = champ_id
                            break
                    else:  # No champion assigned to this role yet
                        blue_team[role_idx] = -1  # Use -1 for empty slots

                    # Find red champion for this role
                    for champ_id, assigned_role in self.state.red_roles.items():
                        if assigned_role == role:
                            red_team[role_idx] = champ_id
                            break
                    else:  # No champion assigned to this role yet
                        red_team[role_idx] = -1  # Use -1 for empty slots

                # Create the draft image
                draft_image = self.visualizer.get_draft_array(blue_team, red_team)

                # Convert numpy array to PNG bytes
                img = Image.fromarray(draft_image.astype("uint8"), "RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()
            return None

    return LoLDraftEnvWithRender
