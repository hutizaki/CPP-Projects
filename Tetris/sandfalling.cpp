#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <cmath>

struct SandParticle {
    sf::Vector2f position;  // For physics-based falling
    sf::Vector2f velocity;  // For physics-based falling
    sf::Color color;
    bool isSettled;         // True when particle has joined the grid
    
    SandParticle(float x, float y) : 
        position(x, y), 
        velocity(0.0f, 0.0f), 
        isSettled(false) {
        // Random sand color variations
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> colorDist(180, 255);
        int r = colorDist(gen);
        int g = colorDist(gen) - 30; // Slightly less green for sand color
        int b = colorDist(gen) - 80; // Even less blue for sand color
        color = sf::Color(r, g, b);
    }
};

class SandGame {
private:
    sf::RenderWindow window;
    sf::RectangleShape particleShape;
    
    // Game settings
    static constexpr int PARTICLE_SIZE = 4;
    static constexpr float GRAVITY = 300.0f;
    static constexpr float AIR_RESISTANCE = 0.98f;
    static constexpr int MAX_FALLING_PARTICLES = 1000;
    
    // Physics-based falling particles
    std::vector<SandParticle> fallingParticles;
    
    // Grid for settled sand
    int gridWidth, gridHeight;
    std::vector<std::vector<bool>> settledGrid; // true = has settled sand, false = empty
    std::vector<std::vector<sf::Color>> settledColors; // color of each settled sand particle
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> spawnSpread;
    std::uniform_int_distribution<int> directionDist;

public:
    SandGame() : 
        window(sf::VideoMode::getDesktopMode(), "Sand Falling Game"),
        gen(rd()),
        spawnSpread(-20.0f, 20.0f),
        directionDist(0, 1) {
        
        window.setFramerateLimit(60);
        
        // Setup particle shape
        particleShape.setSize(sf::Vector2f(PARTICLE_SIZE, PARTICLE_SIZE));
        
        // Initialize grid for settled sand
        gridWidth = static_cast<int>(window.getSize().x / PARTICLE_SIZE);
        gridHeight = static_cast<int>(window.getSize().y / PARTICLE_SIZE);
        settledGrid.resize(gridWidth, std::vector<bool>(gridHeight, false));
        settledColors.resize(gridWidth, std::vector<sf::Color>(gridHeight, sf::Color::Black));
    }
    
    void run() {
        sf::Clock clock;
        
        while (window.isOpen()) {
            float deltaTime = clock.restart().asSeconds();
            
            handleEvents();
            update(deltaTime);
            render();
        }
    }
    
private:
    void handleEvents() {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            if (auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
                    window.close();
                }
            }
            if (auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mousePressed->button == sf::Mouse::Button::Left) {
                    spawnSand(mousePressed->position.x, mousePressed->position.y);
                }
            }
        }
        
        // Continuous sand spawning while mouse is held down
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            spawnSand(mousePos.x, mousePos.y);
        }
    }
    
    void spawnSand(int mouseX, int mouseY) {
        if (fallingParticles.size() < MAX_FALLING_PARTICLES) {
            // Spawn multiple particles with spread for messy dump effect
            for (int i = 0; i < 4; ++i) {
                float offsetX = spawnSpread(gen);
                float offsetY = spawnSpread(gen) * 0.3f;
                
                float x = static_cast<float>(mouseX) + offsetX;
                float y = static_cast<float>(mouseY) + offsetY;
                
                // Ensure spawn position is within window bounds
                if (x >= 0 && x < window.getSize().x && y >= 0 && y < window.getSize().y) {
                    fallingParticles.emplace_back(x, y);
                    // Give particles initial random velocity for more natural look
                    fallingParticles.back().velocity.x = spawnSpread(gen) * 0.5f;
                    fallingParticles.back().velocity.y = std::abs(spawnSpread(gen)) * 0.3f;
                }
            }
        }
    }
    
    void update(float deltaTime) {
        // Update falling particles with physics
        for (auto it = fallingParticles.begin(); it != fallingParticles.end(); ) {
            if (it->isSettled) {
                it = fallingParticles.erase(it);
                continue;
            }
            
            updateFallingParticle(*it, deltaTime);
            ++it;
        }
        
        // Update settled sand grid (pyramid formation)
        updateSettledSand();
    }
    
    void updateFallingParticle(SandParticle& particle, float deltaTime) {
        // Apply gravity
        particle.velocity.y += GRAVITY * deltaTime;
        
        // Apply air resistance
        particle.velocity.x *= AIR_RESISTANCE;
        particle.velocity.y *= AIR_RESISTANCE;
        
        // Update position
        particle.position.x += particle.velocity.x * deltaTime;
        particle.position.y += particle.velocity.y * deltaTime;
        
        // Check if particle should settle
        int gridX = static_cast<int>(particle.position.x / PARTICLE_SIZE);
        int gridY = static_cast<int>(particle.position.y / PARTICLE_SIZE);
        
        // Bounds checking
        if (gridX < 0 || gridX >= gridWidth || gridY >= gridHeight) {
            particle.isSettled = true;
            return;
        }
        
        // Check collision with settled sand or ground
        bool shouldSettle = false;
        
        // Hit the ground
        if (gridY >= gridHeight - 1) {
            shouldSettle = true;
        }
        // Hit settled sand
        else if (gridY >= 0 && settledGrid[gridX][gridY]) {
            shouldSettle = true;
        }
        // Check if there's settled sand below
        else if (gridY + 1 < gridHeight && settledGrid[gridX][gridY + 1]) {
            shouldSettle = true;
        }
        
        if (shouldSettle) {
            // Find a good settling position using grid logic
            settleParticle(particle, gridX, gridY);
        }
    }
    
    void settleParticle(SandParticle& particle, int preferredX, int preferredY) {
        // Clamp to valid grid positions
        preferredX = std::max(0, std::min(gridWidth - 1, preferredX));
        preferredY = std::max(0, std::min(gridHeight - 1, preferredY));
        
        // Try to settle at the preferred position first
        if (!settledGrid[preferredX][preferredY]) {
            settledGrid[preferredX][preferredY] = true;
            settledColors[preferredX][preferredY] = particle.color;
            particle.isSettled = true;
            return;
        }
        
        // Try nearby positions in a small radius
        for (int radius = 1; radius <= 3; ++radius) {
            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    int newX = preferredX + dx;
                    int newY = preferredY + dy;
                    
                    if (newX >= 0 && newX < gridWidth && newY >= 0 && newY < gridHeight) {
                        if (!settledGrid[newX][newY]) {
                            settledGrid[newX][newY] = true;
                            settledColors[newX][newY] = particle.color;
                            particle.isSettled = true;
                            return;
                        }
                    }
                }
            }
        }
        
        // If we can't find anywhere to settle, mark as settled anyway
        particle.isSettled = true;
    }
    
    void updateSettledSand() {
        // Process settled sand for pyramid formation (less frequently for performance)
        static int frameCounter = 0;
        frameCounter++;
        if (frameCounter % 3 != 0) return; // Only update every 3rd frame
        
        // Process sand falling from bottom to top, right to left
        for (int y = gridHeight - 2; y >= 0; --y) {
            for (int x = gridWidth - 1; x >= 0; --x) {
                if (settledGrid[x][y]) {
                    updateSettledSandParticle(x, y);
                }
            }
        }
    }
    
    void updateSettledSandParticle(int x, int y) {
        // Try to fall straight down first
        if (y + 1 < gridHeight && !settledGrid[x][y + 1]) {
            // Move sand down
            settledGrid[x][y + 1] = true;
            settledColors[x][y + 1] = settledColors[x][y];
            settledGrid[x][y] = false;
            return;
        }
        
        // Can't fall straight down, try diagonally
        bool canFallLeft = (x - 1 >= 0 && y + 1 < gridHeight && !settledGrid[x - 1][y + 1]);
        bool canFallRight = (x + 1 < gridWidth && y + 1 < gridHeight && !settledGrid[x + 1][y + 1]);
        
        if (canFallLeft && canFallRight) {
            // Both directions available, choose randomly
            if (directionDist(gen) == 0) {
                // Fall left
                settledGrid[x - 1][y + 1] = true;
                settledColors[x - 1][y + 1] = settledColors[x][y];
                settledGrid[x][y] = false;
            } else {
                // Fall right
                settledGrid[x + 1][y + 1] = true;
                settledColors[x + 1][y + 1] = settledColors[x][y];
                settledGrid[x][y] = false;
            }
        } else if (canFallLeft) {
            // Only left is available
            settledGrid[x - 1][y + 1] = true;
            settledColors[x - 1][y + 1] = settledColors[x][y];
            settledGrid[x][y] = false;
        } else if (canFallRight) {
            // Only right is available
            settledGrid[x + 1][y + 1] = true;
            settledColors[x + 1][y + 1] = settledColors[x][y];
            settledGrid[x][y] = false;
        }
        // If none are available, particle stays where it is (settled)
    }
    
    void render() {
        window.clear(sf::Color::Black);
        
        // Render settled sand from the grid
        for (int x = 0; x < gridWidth; ++x) {
            for (int y = 0; y < gridHeight; ++y) {
                if (settledGrid[x][y]) {
                    particleShape.setPosition(sf::Vector2f(static_cast<float>(x * PARTICLE_SIZE), 
                                                          static_cast<float>(y * PARTICLE_SIZE)));
                    particleShape.setFillColor(settledColors[x][y]);
                    window.draw(particleShape);
                }
            }
        }
        
        // Render falling particles
        for (const auto& particle : fallingParticles) {
            if (!particle.isSettled) {
                particleShape.setPosition(particle.position);
                particleShape.setFillColor(particle.color);
                window.draw(particleShape);
            }
        }
        
        window.display();
    }
};

int main() {
    SandGame game;
    game.run();
    return 0;
}
