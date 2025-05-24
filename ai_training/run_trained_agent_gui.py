# run_agent_gui.py
# -------------------------------------------------------------------------
# Demonstrate a Maskable-PPO agent in the Seven Wonders environment.
# The interface is deliberately minimal: highlight the move the agent
# intends to take, execute it on SPACE, and keep the usual HUD.

import os
import sys
import pygame
import torch
from sb3_contrib import MaskablePPO

# Project-local imports ----------------------------------------------------
from .sw_gym_env import SevenWondersEnv               # gym wrapper used in training
import config                                         # rows, cols, asset paths, constants

# --- files / hyper --------------------------------------------------------
MODEL_PATH = "ai_training/models/7wonders_ppo_v4/7wonders_ppo_v4_final.zip"     # adjust if necessary
TILE_SIZE  = 50
GRID_MARGIN = 2
SCREEN_WIDTH  = 10 * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN
SCREEN_HEIGHT = 10 * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN + 100
FPS = 60

# --- colours --------------------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT_ACTION = (255, 255, 0, 128)      # where the agent *will* move
BLUE_MASK = (0, 0, 255, 64)                # any legal move

# -------------------------------------------------------------------------
class AgentGUI:
    def __init__(self):
        # pygame surface ---------------------------------------------------
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("7 Wonders – RL agent")
        self.clock = pygame.time.Clock()

        # env / model ------------------------------------------------------
        self.env, self.obs = self._build_env()
        self.model = MaskablePPO.load(MODEL_PATH, device="cpu")
        self.pending_action_idx, self.pending_swap = self._policy_action()
        self.last_reward = 0.0

        # graphics ---------------------------------------------------------
        self.assets = {}
        self._load_assets()

    # ---------------------------------------------------------------------
    @staticmethod
    def _build_env():
        env = SevenWondersEnv(level=config.LEVEL_1)
        obs, _info = env.reset()
        return env, obs

    # ---------------------------------------------------------------------
    def _policy_action(self):
        """Ask the policy for its **next** move, return (idx, swap-tuple)."""
        idx, _ = self.model.predict(
            self.obs, deterministic=True, action_masks=self.env.action_masks()
        )
        return idx, self.env._decode_action(idx)

    # ---------------------------------------------------------------------
    def _load_assets(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_dir = os.path.join(root, "capture_dataset", "elements")

        def load_scaled(name, filename):
            path = os.path.join(asset_dir, filename)
            if os.path.exists(path):
                img = pygame.image.load(path)
                self.assets[name] = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))

        # gems
        for i in range(8):
            for ext in (".png", ".jpg", ".gif"):
                if f"gem_{i}" in self.assets:
                    break
                load_scaled(f"gem_{i}", f"gem_{i}{ext}")

        # bonuses
        for i in range(3):
            load_scaled(f"bonus_{i}", f"bonus_{i}.png")

        # backgrounds & fragment
        bg_files = {
            "stone": "stone.png",
            "stone_shield": "stone_shield.png",
            "empty": "empty.png",
            "fragment": "bloc.gif",
        }
        for key, fn in bg_files.items():
            load_scaled(key, fn)

    # ---------------------------------------------------------------------
    #                           RENDERING
    # ---------------------------------------------------------------------
    def draw_board(self):
        g = self.env.sim                  # underlying simulator

        self.screen.fill(WHITE)

        # list of legal swaps for current state
        valid_swaps = g.get_valid_swaps()

        # board tiles ------------------------------------------------------
        for r in range(g.rows):
            for c in range(g.cols):
                if not g.mask[r, c]:
                    continue

                x = c * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN
                y = r * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN

                # background
                bg = {
                    g.BG_STONE: "stone",
                    g.BG_SHIELD: "stone_shield",
                }.get(g.background[r, c], "empty")
                self.screen.blit(self.assets[bg], (x, y))

                # content
                content = g.content[r, c]
                if content == g.EMPTY:
                    pass
                elif content == g.FRAGMENT:
                    self.screen.blit(self.assets["fragment"], (x, y))
                elif g.BONUS_0 <= content <= g.BONUS_2:
                    self.screen.blit(
                        self.assets[f"bonus_{content - g.BONUS_0}"], (x, y)
                    )
                elif g.GEM_START_IDX <= content <= g.GEM_END_IDX:
                    self.screen.blit(
                        self.assets[f"gem_{content - g.GEM_START_IDX}"], (x, y)
                    )

                # mark any swap that is legal
                if any((r, c) in sw for sw in valid_swaps):
                    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    surf.fill(BLUE_MASK)
                    self.screen.blit(surf, (x, y))

                # highlight the agent's chosen swap
                if (r, c) in self.pending_swap:
                    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    surf.fill(HIGHLIGHT_ACTION)
                    self.screen.blit(surf, (x, y))

        # HUD --------------------------------------------------------------
        font = pygame.font.Font(None, 24)
        def hud(text, x_frac):
            txt = font.render(text, True, BLACK)
            self.screen.blit(txt, (SCREEN_WIDTH * x_frac, SCREEN_HEIGHT - 40))

        hud(f"Score {g.score:.2f}",            0.00)
        hud(f"Stones {g.stones_cleared}/{g.initial_stones}", 0.25)
        hud(f"Fragments {g.fragments_on_board}", 0.50)
        hud(f"Steps {g.step_count}",           0.75)

        # bottom line: agent's move + reward
        moves_font = pygame.font.Font(None, 18)
        move_str = f"→ agent intends: {self.pending_swap[0]} ↔ {self.pending_swap[1]}"
        reward_str = f" | last reward: {self.last_reward:+.3f}"
        txt = moves_font.render(move_str + reward_str, True, BLACK)
        self.screen.blit(txt, (10, SCREEN_HEIGHT - 70))

    # ---------------------------------------------------------------------
    #                           MAIN LOOP
    # ---------------------------------------------------------------------
    def run(self):
        running = True
        done = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE and not done:
                        # perform the pending swap -----------------------------------
                        self.obs, self.last_reward, done, _trunc, _info = self.env.step(
                            self.pending_action_idx
                        )

                        if done:
                            print(f"Episode finished: score={self.env.sim.score:.2f}")
                        else:
                            # ask for the next move
                            self.pending_action_idx, self.pending_swap = self._policy_action()

            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    AgentGUI().run()
