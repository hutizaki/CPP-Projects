#include <SFML/Graphics.hpp>
#include <chrono>
#include <random>
#include <cstdint>

using namespace std;
using namespace sf;

float WIDTH = 1000;
float HEIGHT = 700;

int numAgents = 1000;
float SIZE = 2;
int JUMP = 3;

float STEP_TIME = 0.01f; // seconds per jump
float stepTimer = 0.f;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 engine(seed);
std::uniform_int_distribution<int> dist(0, 3);

struct Vector {
    float x, y;
};

struct Agent {
    Color pathColor;
    Vector2f pos;
    int behindDirection;

    Agent(Color c) : pathColor(c), pos(WIDTH / 2.f, HEIGHT / 2.f) {}
};

int boundsCheck(Agent& a) {
    int r = dist(engine);
    Vector2f currPos = a.pos;
    if ((currPos.x - JUMP <= JUMP*SIZE && r == 3) || (currPos.x + JUMP*SIZE >= WIDTH - JUMP*SIZE && r == 1) ||
        (currPos.y - JUMP <= JUMP*SIZE && r == 0) || (currPos.y + JUMP*SIZE >= HEIGHT - JUMP*SIZE && r == 2)) 
    {
        return -1;    
    }
    return r;
}

Vector randomV(Agent &a) {
    int x = boundsCheck(a);
    while (x == -1) {
        x = boundsCheck(a);
    }
    while (x == a.behindDirection) {
      x = dist(engine);
    }
    switch (x) {
    case 0:
      a.behindDirection = 2;
      return {0.f, -SIZE}; // UP
    case 1:
      a.behindDirection = 3;
      return {SIZE, 0.f}; // RIGHT
    case 2:
      a.behindDirection = 0;
      return {0.f, SIZE}; // DOWN
    case 3:
      a.behindDirection = 1;
      return {-SIZE, 0.f}; // LEFT
    }
    return {0.f, 0.f};
}

void fillRect(RenderTexture &canvas, float x, float y, Color color) {
    RectangleShape rect({SIZE, SIZE});
    rect.setPosition({x, y});
    rect.setFillColor(color);
    canvas.draw(rect);
}

void moveAgent(Agent &a, RenderTexture &canvas) {
    Vector v = randomV(a);
    for (int i = 0; i < JUMP; i++) {
    a.pos.x += v.x;
    a.pos.y += v.y;

    // draw onto the persistent canvas
    fillRect(canvas, a.pos.x, a.pos.y, a.pathColor);
    }
}

static float clamp01(float x) {
  if (x < 0.f)
    return 0.f;
  if (x > 1.f)
    return 1.f;
  return x;
}

static float hue2rgb(float p, float q, float t) {
  if (t < 0.f)
    t += 1.f;
  if (t > 1.f)
    t -= 1.f;
  if (t < 1.f / 6.f)
    return p + (q - p) * 6.f * t;
  if (t < 1.f / 2.f)
    return q;
  if (t < 2.f / 3.f)
    return p + (q - p) * (2.f / 3.f - t) * 6.f;
  return p;
}

static sf::Color HSLtoRGB(float h, float s, float l) {
  // h,s,l are in [0,1]
  h = h - std::floor(h); // wrap hue
  s = clamp01(s);
  l = clamp01(l);

  float r, g, b;

  if (s == 0.f) {
    // gray
    r = g = b = l;
  } else {
    float q = (l < 0.5f) ? (l * (1.f + s)) : (l + s - l * s);
    float p = 2.f * l - q;
    r = hue2rgb(p, q, h + 1.f / 3.f);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1.f / 3.f);
  }

  return sf::Color(static_cast<std::uint8_t>(r * 255.f),
                   static_cast<std::uint8_t>(g * 255.f),
                   static_cast<std::uint8_t>(b * 255.f));
}

// Build a "random-ish" HSL color using ONLY dist(engine) (0..3).
static sf::Color randomHSLColor(std::mt19937 &engine,
                                std::uniform_int_distribution<int> &dist) {
  // Hue: use 8 bits from dist => 0..255 then /256
  int hBits = 0;
  for (int i = 0; i < 4; i++) {          // 4 draws * 2 bits = 8 bits
    hBits = (hBits << 2) | dist(engine); // dist gives 0..3 (2 bits)
  }
  float h = (hBits / 256.f); // [0,1)

  // Saturation: 0.60..1.00 (use 4 bits => 0..15)
  int sBits = 0;
  for (int i = 0; i < 2; i++) { // 2 draws -> 4 bits
    sBits = (sBits << 2) | dist(engine);
  }
  float s = 0.60f + (sBits / 15.f) * 0.40f; // [0.60, 1.00]

  // Lightness: 0.35..0.65 (use 4 bits => 0..15)
  int lBits = 0;
  for (int i = 0; i < 2; i++) {
    lBits = (lBits << 2) | dist(engine);
  }
  float l = 0.35f + (lBits / 15.f) * 0.30f; // [0.35, 0.65]

  return HSLtoRGB(h, s, l);
}

int main() {
    RenderWindow window(VideoMode({(unsigned)WIDTH, (unsigned)HEIGHT}),
                        "Random Walk");
    window.setFramerateLimit(60);

    RenderTexture canvas;
    canvas.resize({(unsigned)WIDTH, (unsigned)HEIGHT});
    canvas.clear(Color::Black); // clear ONCE at the start

    vector<Agent> agents;
    agents.reserve(numAgents);

    for (int i = 0; i < numAgents; i++) {
      Color c = randomHSLColor(engine, dist);
      agents.emplace_back(c);
    }

    canvas.display();

    Clock clock;

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<Event::Closed>())
            window.close();
        }

        float dt = clock.restart().asSeconds();
        stepTimer += dt;

        while (stepTimer >= STEP_TIME) {
            stepTimer -= STEP_TIME;
            for (int i = 0; i < numAgents; i++) {
                moveAgent(agents[i], canvas);
            }
        }

        // finalize canvas updates for this frame
        canvas.display();

        // draw canvas to window every frame
        window.clear();
        Sprite s(canvas.getTexture());
        window.draw(s);
        window.display();
    }
}
