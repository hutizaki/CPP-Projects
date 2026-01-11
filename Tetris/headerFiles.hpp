#include <SFML/Graphics.hpp>

class squareShape {
    public:
        squareShape();
        void draw(sf::RenderWindow& window);
        void move(int newX, int newY);
    private:
        sf::RectangleShape block1, block2, block3, block4;
        int x;
        int y;
        sf::Color color;
};