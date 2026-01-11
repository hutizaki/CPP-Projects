#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <vector>
#include <random>
#include <iostream>

// Game constants
const int GRID_WIDTH = 10;
const int GRID_HEIGHT = 20;
const int CELL_SIZE = 30;
const int WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + 200; // Extra space for UI
const int WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 100;

// Tetris piece types (7 standard pieces)
enum PieceType { I, O, T, S, Z, J, L };

// Colors for each piece type
sf::Color pieceColors[7] = {
    sf::Color::Cyan,    // I
    sf::Color::Yellow,  // O
    sf::Color::Magenta, // T
    sf::Color::Green,   // S
    sf::Color::Red,     // Z
    sf::Color::Blue,    // J
    sf::Color(255, 165, 0) // L (Orange)
};

// Tetris piece shapes (4x4 grid for each piece and rotation)
int pieces[7][4][4][4] = {
    // I piece
    {{{0,0,0,0},{1,1,1,1},{0,0,0,0},{0,0,0,0}},
     {{0,0,1,0},{0,0,1,0},{0,0,1,0},{0,0,1,0}},
     {{0,0,0,0},{0,0,0,0},{1,1,1,1},{0,0,0,0}},
     {{0,1,0,0},{0,1,0,0},{0,1,0,0},{0,1,0,0}}},
    // O piece
    {{{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}}},
    // T piece
    {{{0,0,0,0},{0,1,0,0},{1,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,0,0},{0,1,1,0},{0,1,0,0}},
     {{0,0,0,0},{0,0,0,0},{1,1,1,0},{0,1,0,0}},
     {{0,0,0,0},{0,1,0,0},{1,1,0,0},{0,1,0,0}}},
    // S piece
    {{{0,0,0,0},{0,1,1,0},{1,1,0,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,0,0},{0,1,1,0},{0,0,1,0}},
     {{0,0,0,0},{0,0,0,0},{0,1,1,0},{1,1,0,0}},
     {{0,0,0,0},{1,0,0,0},{1,1,0,0},{0,1,0,0}}},
    // Z piece
    {{{0,0,0,0},{1,1,0,0},{0,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,0,1,0},{0,1,1,0},{0,1,0,0}},
     {{0,0,0,0},{0,0,0,0},{1,1,0,0},{0,1,1,0}},
     {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,0,0,0}}},
    // J piece
    {{{0,0,0,0},{1,0,0,0},{1,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,1,0},{0,1,0,0},{0,1,0,0}},
     {{0,0,0,0},{0,0,0,0},{1,1,1,0},{0,0,1,0}},
     {{0,0,0,0},{0,1,0,0},{0,1,0,0},{1,1,0,0}}},
    // L piece
    {{{0,0,0,0},{0,0,1,0},{1,1,1,0},{0,0,0,0}},
     {{0,0,0,0},{0,1,0,0},{0,1,0,0},{0,1,1,0}},
     {{0,0,0,0},{0,0,0,0},{1,1,1,0},{1,0,0,0}},
     {{0,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,0,0}}}
};

class TetrisGame {
private:
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<sf::Color>> gridColors;
    int currentPiece;
    int currentRotation;
    int currentX, currentY;
    int nextPiece;
    int score;
    int level;
    int linesCleared;
    bool gameOver;
    
    std::mt19937 rng;
    std::uniform_int_distribution<int> pieceDist;
    
    sf::Clock fallTimer;
    sf::Clock inputTimer;
    float fallSpeed;
    float inputDelay;
    
public:
    TetrisGame() : grid(GRID_HEIGHT, std::vector<int>(GRID_WIDTH, 0)),
                   gridColors(GRID_HEIGHT, std::vector<sf::Color>(GRID_WIDTH, sf::Color::Black)),
                   score(0), level(1), linesCleared(0), gameOver(false),
                   fallSpeed(1.0f), inputDelay(0.1f),
                   rng(std::random_device{}()),
                   pieceDist(0, 6) {
        spawnNewPiece();
        nextPiece = pieceDist(rng);
    }
    
    void spawnNewPiece() {
        currentPiece = (currentPiece == -1) ? pieceDist(rng) : nextPiece;
        nextPiece = pieceDist(rng);
        currentRotation = 0;
        currentX = GRID_WIDTH / 2 - 2;
        currentY = 0;
        
        if (isColliding(currentX, currentY, currentPiece, currentRotation)) {
            gameOver = true;
        }
    }
    
    bool isColliding(int x, int y, int piece, int rotation) {
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                if (pieces[piece][rotation][py][px]) {
                    int newX = x + px;
                    int newY = y + py;
                    
                    if (newX < 0 || newX >= GRID_WIDTH || newY >= GRID_HEIGHT) {
                        return true;
                    }
                    if (newY >= 0 && grid[newY][newX]) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    void lockPiece() {
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                if (pieces[currentPiece][currentRotation][py][px]) {
                    int gridX = currentX + px;
                    int gridY = currentY + py;
                    if (gridY >= 0) {
                        grid[gridY][gridX] = 1;
                        gridColors[gridY][gridX] = pieceColors[currentPiece];
                    }
                }
            }
        }
        
        clearLines();
        spawnNewPiece();
    }
    
    void clearLines() {
        int linesFound = 0;
        for (int y = GRID_HEIGHT - 1; y >= 0; y--) {
            bool fullLine = true;
            for (int x = 0; x < GRID_WIDTH; x++) {
                if (!grid[y][x]) {
                    fullLine = false;
                    break;
                }
            }
            
            if (fullLine) {
                // Remove the line
                grid.erase(grid.begin() + y);
                gridColors.erase(gridColors.begin() + y);
                // Add empty line at top
                grid.insert(grid.begin(), std::vector<int>(GRID_WIDTH, 0));
                gridColors.insert(gridColors.begin(), std::vector<sf::Color>(GRID_WIDTH, sf::Color::Black));
                linesFound++;
                y++; // Check same line again
            }
        }
        
        if (linesFound > 0) {
            linesCleared += linesFound;
            // Scoring: 1 line = 100, 2 lines = 300, 3 lines = 500, 4 lines = 800
            int lineScore[] = {0, 100, 300, 500, 800};
            score += lineScore[linesFound] * level;
            
            // Increase level every 10 lines
            level = (linesCleared / 10) + 1;
            fallSpeed = 1.0f - (level - 1) * 0.1f;
            if (fallSpeed < 0.1f) fallSpeed = 0.1f;
        }
    }
    
    void handleInput() {
        if (gameOver) return;
        
        if (inputTimer.getElapsedTime().asSeconds() >= inputDelay) {
            bool moved = false;
            
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left) || 
                sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) {
                if (!isColliding(currentX - 1, currentY, currentPiece, currentRotation)) {
                    currentX--;
                    moved = true;
                }
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right) || 
                sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) {
                if (!isColliding(currentX + 1, currentY, currentPiece, currentRotation)) {
                    currentX++;
                    moved = true;
                }
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down) || 
                sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) {
                if (!isColliding(currentX, currentY + 1, currentPiece, currentRotation)) {
                    currentY++;
                    score++; // Bonus for soft drop
                    moved = true;
                }
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up) || 
                sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) {
                int newRotation = (currentRotation + 1) % 4;
                if (!isColliding(currentX, currentY, currentPiece, newRotation)) {
                    currentRotation = newRotation;
                    moved = true;
                }
            }
            
            if (moved) {
                inputTimer.restart();
            }
        }
    }
    
    void update() {
        if (gameOver) return;
        
        // Automatic falling
        if (fallTimer.getElapsedTime().asSeconds() >= fallSpeed) {
            if (!isColliding(currentX, currentY + 1, currentPiece, currentRotation)) {
                currentY++;
            } else {
                lockPiece();
            }
            fallTimer.restart();
        }
    }
    
    void draw(sf::RenderWindow& window) {
        // Draw grid background
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE - 1, CELL_SIZE - 1));
        
        // Draw placed pieces
        for (int y = 0; y < GRID_HEIGHT; y++) {
            for (int x = 0; x < GRID_WIDTH; x++) {
                cell.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                if (grid[y][x]) {
                    cell.setFillColor(gridColors[y][x]);
                } else {
                    cell.setFillColor(sf::Color(50, 50, 50)); // Dark grid
                }
                window.draw(cell);
            }
        }
        
        // Draw current falling piece
        if (!gameOver) {
            cell.setFillColor(pieceColors[currentPiece]);
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    if (pieces[currentPiece][currentRotation][py][px]) {
                        int drawX = (currentX + px) * CELL_SIZE;
                        int drawY = (currentY + py) * CELL_SIZE;
                        if (currentY + py >= 0) { // Don't draw above grid
                            cell.setPosition(sf::Vector2f(drawX, drawY));
                            window.draw(cell);
                        }
                    }
                }
            }
        }
        
        // Draw next piece preview
        cell.setFillColor(pieceColors[nextPiece]);
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                if (pieces[nextPiece][0][py][px]) {
                    int drawX = (GRID_WIDTH + 1 + px) * CELL_SIZE;
                    int drawY = (2 + py) * CELL_SIZE;
                    cell.setPosition(sf::Vector2f(drawX, drawY));
                    window.draw(cell);
                }
            }
        }
    }
    
    void drawUI(sf::RenderWindow& window, sf::Font& font) {
        sf::Text text(font, "");
        text.setFont(font);
        text.setCharacterSize(20);
        text.setFillColor(sf::Color::White);
        
        // Score
        text.setString("Score: " + std::to_string(score));
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 50));
        window.draw(text);
        
        // Level
        text.setString("Level: " + std::to_string(level));
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 80));
        window.draw(text);
        
        // Lines
        text.setString("Lines: " + std::to_string(linesCleared));
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 110));
        window.draw(text);
        
        // Next piece label
        text.setString("Next:");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 20));
        window.draw(text);
        
        // Game Over
        if (gameOver) {
            text.setCharacterSize(30);
            text.setFillColor(sf::Color::Red);
            text.setString("GAME OVER");
            text.setPosition(sf::Vector2f(50, GRID_HEIGHT * CELL_SIZE / 2));
            window.draw(text);
            
            text.setCharacterSize(20);
            text.setFillColor(sf::Color::White);
            text.setString("Press R to restart");
            text.setPosition(sf::Vector2f(50, GRID_HEIGHT * CELL_SIZE / 2 + 40));
            window.draw(text);
        }
        
        // Controls
        text.setCharacterSize(16);
        text.setString("Controls:");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 200));
        window.draw(text);
        
        text.setCharacterSize(14);
        text.setString("WASD/Arrows");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 220));
        window.draw(text);
        
        text.setString("Up: Rotate");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 240));
        window.draw(text);
        
        text.setString("Down: Soft Drop");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 260));
        window.draw(text);
        
        text.setString("R: Restart");
        text.setPosition(sf::Vector2f(GRID_WIDTH * CELL_SIZE + 10, 280));
        window.draw(text);
    }
    
    void restart() {
        grid = std::vector<std::vector<int>>(GRID_HEIGHT, std::vector<int>(GRID_WIDTH, 0));
        gridColors = std::vector<std::vector<sf::Color>>(GRID_HEIGHT, std::vector<sf::Color>(GRID_WIDTH, sf::Color::Black));
        score = 0;
        level = 1;
        linesCleared = 0;
        gameOver = false;
        fallSpeed = 1.0f;
        currentPiece = -1; // Force new piece generation
        spawnNewPiece();
        nextPiece = pieceDist(rng);
        fallTimer.restart();
    }
    
    bool isGameOver() const { return gameOver; }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(WINDOW_WIDTH, WINDOW_HEIGHT)), "Tetris");
    window.setFramerateLimit(60);
    
    // Load font
    sf::Font font;
    if (!font.openFromFile("arial.ttf")) {
        std::cout << "Warning: Could not load font. Text will not display properly." << std::endl;
    }
    
    // Load music
    sf::Music music;
    if (music.openFromFile("nice_music.ogg")) {
        music.setLooping(true);
        music.play();
    }
    
    TetrisGame game;
    
    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::R) {
                    game.restart();
                }
            }
        }
        
        game.handleInput();
        game.update();
        
        window.clear(sf::Color::Black);
        game.draw(window);
        game.drawUI(window, font);
        window.display();
    }
    
    return 0;
}