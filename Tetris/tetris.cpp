#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include "headerFiles.hpp"

int main() {

    sf::Music music("nice_music.ogg");
    music.play();

    const int cell = 30, cols = 10, rows = 20;
    sf::RenderWindow window(sf::VideoMode({static_cast<unsigned int>(cols*cell), static_cast<unsigned int>(rows*cell)}), "Tetris (SFML)");
    window.setFramerateLimit(60);
    
    squareShape block;

    int px = 0, py = 0; // grid position
    bool blockLocked = false; // Track if current block is locked

    // Timing for input control
    sf::Clock inputClock;
    float inputDelay = 0.1f; // seconds between moves
    float inputTimer = 0.f;

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        // Input with timing control
        float dt = inputClock.restart().asSeconds();
        inputTimer += dt;
        
        if (inputTimer >= inputDelay && !blockLocked) {
            bool moved = false;
            // Arrow keys move the cyan square
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))  { py = std::max(0, py-1); moved = true; }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))  { px = std::max(0, px-1); moved = true; }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { px = std::min(cols-2, px+1); moved = true; }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))  { 
                if (py < rows-2) { // Only move down if not at bottom
                    py = std::min(rows-2, py+1); 
                    moved = true; 
                }
            }
            
            if (moved) {
                inputTimer = 0.f; // Reset timer only when a move was made
                block.move(px, py);
                
                // Check if block has reached the bottom and should be locked
                if (py == rows-2) {
                    blockLocked = true;
                }
            }
        }

        // Draw
        window.clear(sf::Color(0,0,0));

        block.draw(window);
        
        window.display();
    }
    return 0;
}
