import pygame
import os
import sys
from .game_simulator import SevenWondersSimulator, LEVEL_1
import config

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 50
GRID_MARGIN = 2
SCREEN_WIDTH = 10 * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN
SCREEN_HEIGHT = 10 * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN + 100  # Extra space for score
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
HIGHLIGHT = (255, 255, 0, 128)  # Yellow with transparency
VALID_MOVE_HIGHLIGHT = (0, 0, 255, 64)  # Blue with transparency

class SevenWondersGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("7 Wonders")
        self.clock = pygame.time.Clock()
        self.game = SevenWondersSimulator(level=config.LEVEL_6)
        self.selected_tile = None
        self.load_assets()
        self.valid_moves = []  # Store valid moves
        
    def load_assets(self):
        """Load all game assets"""
        self.assets = {}
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_dir = os.path.join(project_root, 'capture_dataset', 'elements')
        
        # Load gems
        for i in range(8):
            # Try different extensions
            for ext in ['.jpg', '.gif', '.png']:
                path = os.path.join(asset_dir, f'gem_{i}{ext}')
                if os.path.exists(path):
                    try:
                        self.assets[f'gem_{i}'] = pygame.image.load(path)
                        self.assets[f'gem_{i}'] = pygame.transform.scale(self.assets[f'gem_{i}'], (TILE_SIZE, TILE_SIZE))
                        break
                    except pygame.error as e:
                        print(f"Warning: Could not load {path}: {e}")
            
        # Load bonuses
        for i in range(3):
            path = os.path.join(asset_dir, f'bonus_{i}.png')
            if os.path.exists(path):
                try:
                    self.assets[f'bonus_{i}'] = pygame.image.load(path)
                    self.assets[f'bonus_{i}'] = pygame.transform.scale(self.assets[f'bonus_{i}'], (TILE_SIZE, TILE_SIZE))
                except pygame.error as e:
                    print(f"Warning: Could not load {path}: {e}")
            
        # Load backgrounds
        background_assets = {
            'stone': 'stone.png',
            'stone_shield': 'stone_shield.png',
            'empty': 'empty.png',
            'fragment': 'bloc.gif'
        }
        
        for name, filename in background_assets.items():
            path = os.path.join(asset_dir, filename)
            if os.path.exists(path):
                try:
                    self.assets[name] = pygame.image.load(path)
                    self.assets[name] = pygame.transform.scale(self.assets[name], (TILE_SIZE, TILE_SIZE))
                except pygame.error as e:
                    print(f"Warning: Could not load {path}: {e}")
            
        # Verify all required assets are loaded
        required_assets = [f'gem_{i}' for i in range(8)] + \
                         [f'bonus_{i}' for i in range(3)] + \
                         ['stone', 'stone_shield', 'empty', 'fragment']
        
        missing_assets = [asset for asset in required_assets if asset not in self.assets]
        if missing_assets:
            print(f"Warning: Missing assets: {missing_assets}")
            print(f"Asset directory: {asset_dir}")
            print("Available files:", os.listdir(asset_dir))
        
    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(WHITE)
        
        # Update valid moves
        self.valid_moves = self.game.get_valid_swaps()
        
        # Draw grid
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                if not self.game.mask[r, c]:
                    continue  # Skip holes
                    
                # Calculate position
                x = c * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN
                y = r * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN
                
                # Draw background
                if self.game.background[r, c] == self.game.BG_STONE:
                    self.screen.blit(self.assets['stone'], (x, y))
                elif self.game.background[r, c] == self.game.BG_SHIELD:
                    self.screen.blit(self.assets['stone_shield'], (x, y))
                else:
                    self.screen.blit(self.assets['empty'], (x, y))
                
                # Draw content
                content = self.game.content[r, c]
                if content == self.game.EMPTY:
                    continue
                elif content == self.game.FRAGMENT:
                    self.screen.blit(self.assets['fragment'], (x, y))
                elif self.game.BONUS_0 <= content <= self.game.BONUS_2:
                    self.screen.blit(self.assets[f'bonus_{content - self.game.BONUS_0}'], (x, y))
                elif self.game.GEM_START_IDX <= content <= self.game.GEM_END_IDX:
                    self.screen.blit(self.assets[f'gem_{content - self.game.GEM_START_IDX}'], (x, y))
                
                # Draw selection highlight
                if self.selected_tile == (r, c):
                    highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    highlight.fill(HIGHLIGHT)
                    self.screen.blit(highlight, (x, y))
                
                # Draw valid move highlight
                for move in self.valid_moves:
                    if (r, c) in move:
                        highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                        highlight.fill(VALID_MOVE_HIGHLIGHT)
                        self.screen.blit(highlight, (x, y))
        
        # Draw score
        font = pygame.font.Font(None, 24)  # Smaller font size
        score_text = font.render(f'Score: {self.game.score:.2f}', True, BLACK)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 40))
        
        # Draw stones cleared
        stones_text = font.render(f'Stones: {self.game.stones_cleared}/{self.game.initial_stones}', True, BLACK)
        self.screen.blit(stones_text, (SCREEN_WIDTH * 0.25, SCREEN_HEIGHT - 40))
        
        # Draw fragments
        fragments_text = font.render(f'Fragments: {self.game.fragments_on_board}', True, BLACK)
        self.screen.blit(fragments_text, (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT - 40))

        # Draw steps
        steps_text = font.render(f'Steps: {self.game.step_count}', True, BLACK)
        self.screen.blit(steps_text, (SCREEN_WIDTH * 0.75, SCREEN_HEIGHT - 40))

        # Draw available moves with wrapping
        moves_font = pygame.font.Font(None, 18)  # Even smaller font for moves
        moves = self.game.get_valid_swaps()
        moves_text = f'Moves: {len(moves)}'
        moves_surface = moves_font.render(moves_text, True, BLACK)
        self.screen.blit(moves_surface, (10, SCREEN_HEIGHT - 100))

        # Draw moves details with wrapping
        if moves:
            line_height = 20
            max_width = SCREEN_WIDTH - 20  # Leave some margin
            current_y = SCREEN_HEIGHT - 80
            current_x = 10
            current_line = ""

            for move in moves:
                move_str = f"({move[0][0]},{move[0][1]})-({move[1][0]},{move[1][1]})"
                test_line = current_line + " " + move_str if current_line else move_str
                test_surface = moves_font.render(test_line, True, BLACK)
                
                if test_surface.get_width() > max_width:
                    # Draw current line and start new one
                    if current_line:
                        moves_surface = moves_font.render(current_line, True, BLACK)
                        self.screen.blit(moves_surface, (current_x, current_y))
                        current_y += line_height
                    current_line = move_str
                else:
                    current_line = test_line

            # Draw the last line if there is one
            if current_line:
                moves_surface = moves_font.render(current_line, True, BLACK)
                self.screen.blit(moves_surface, (current_x, current_y))
        
    def handle_click(self, pos):
        """Handle mouse click"""
        x, y = pos
        if y >= SCREEN_HEIGHT - 100:  # Click in score area
            return
            
        # Convert screen coordinates to grid coordinates
        c = x // (TILE_SIZE + GRID_MARGIN)
        r = y // (TILE_SIZE + GRID_MARGIN)
        
        if not (0 <= r < self.game.rows and 0 <= c < self.game.cols):
            return
            
        if not self.game.mask[r, c]:
            return  # Clicked on a hole
            
        if self.selected_tile is None:
            # First tile selection
            if self.game.content[r, c] != self.game.EMPTY and self.game.content[r, c] != self.game.FRAGMENT:
                self.selected_tile = (r, c)
        else:
            # Second tile selection - try to swap
            if self.selected_tile != (r, c):
                # Check if the swap is valid
                valid_swaps = self.game.get_valid_swaps()
                swap = (self.selected_tile, (r, c))
                if swap in valid_swaps or (swap[1], swap[0]) in valid_swaps:
                    # Execute the swap
                    new_state, reward, done = self.game.step(swap)
                    if done:
                        print(f"Game Over! Final Score: {self.game.score}")
                        pygame.quit()
                        sys.exit()
            self.selected_tile = None
            
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    game = SevenWondersGUI()
    game.run() 