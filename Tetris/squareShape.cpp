#include "headerFiles.hpp"

squareShape::squareShape() {
    x = 0;
    y = 0;
    block1.setSize(sf::Vector2f(30-2, 30-2));
    block1.setFillColor(sf::Color(255, 255, 0));
    block2 = block1;
    block3 = block1;
    block4 = block1;
    block1.setPosition(sf::Vector2f(x * 30 + 1,  y * 30 + 1));
    block2.setPosition(sf::Vector2f(x * 30 + 31, y * 30 + 1));
    block3.setPosition(sf::Vector2f(x * 30 + 1,  y * 30 + 31));
    block4.setPosition(sf::Vector2f(x * 30 + 31, y * 30 + 31));
}

void squareShape::move(int newX, int newY) {
    x = newX;
    y = newY;
    block1.setPosition(sf::Vector2f(x * 30 + 1,  y * 30 + 1));
    block2.setPosition(sf::Vector2f(x * 30 + 31, y * 30 + 1));
    block3.setPosition(sf::Vector2f(x * 30 + 1,  y * 30 + 31));
    block4.setPosition(sf::Vector2f(x * 30 + 31, y * 30 + 31));
}

void squareShape::draw(sf::RenderWindow& window) {
    window.draw(block1);
    window.draw(block2);
    window.draw(block3);
    window.draw(block4);
}