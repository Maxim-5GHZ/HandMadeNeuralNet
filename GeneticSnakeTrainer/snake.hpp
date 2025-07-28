#include <deque>
#include <random>
#include <stdexcept>
#include <ncurses.h>
#include <iostream>
#include <vector>
#include <cmath>

struct Position {
    int x, y;
    Position(int col = 0, int row = 0) : x(col), y(row) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

class SnakeGame {
private:
    int width, height;
    Position food;
    std::deque<Position> snake;
    int direction; 
    bool game_over;
    int score;
    std::mt19937 gen;
    std::uniform_int_distribution<> dist_x;
    std::uniform_int_distribution<> dist_y;
    int step_count;
    int max_steps;

    void place_food() {
        int attempts = 0;
        const int max_attempts = (width - 2) * (height - 2);
        
        while (attempts++ < max_attempts) {
            food.x = dist_x(gen);
            food.y = dist_y(gen);
            
            bool on_snake = false;
            for (const auto& segment : snake) {
                if (food == segment) {
                    on_snake = true;
                    break;
                }
            }
            
            if (!on_snake) return;
        }
        
        game_over = true;
    }

    void process_input(int key) {
        switch (key) {
            case KEY_UP:
                if (direction != 2) direction = 0;
                break;
            case KEY_RIGHT:
                if (direction != 3) direction = 1;
                break;
            case KEY_DOWN:
                if (direction != 0) direction = 2;
                break;
            case KEY_LEFT:
                if (direction != 1) direction = 3;
                break;
            case 'q':
                game_over = true;
                break;
            case 'p':
                while (getch() != 'p') {} 
                break;
        }
    }

    void update() {
        Position head = snake.front();
        
        switch (direction) {
            case 0: head.y--; break;
            case 1: head.x++; break;
            case 2: head.y++; break;
            case 3: head.x--; break;
        }
        
        if (head.x <= 0 || head.x >= width-1 || head.y <= 0 || head.y >= height-1) {
            game_over = true;
            return;
        }
        
        for (auto it = snake.begin() + 1; it != snake.end(); ++it) {
            if (head == *it) {
                game_over = true;
                return;
            }
        }

        snake.push_front(head);

        if (head == food) {
            score += 10;
            place_food();
        } else {
            snake.pop_back();
        }
    }

    void draw() const {
        clear();
        
        for (int i = 0; i < width; i++) {
            mvaddch(0, i, '#');
            mvaddch(height-1, i, '#');
        }
        for (int i = 0; i < height; i++) {
            mvaddch(i, 0, '#');
            mvaddch(i, width-1, '#');
        }

        for (const auto& segment : snake) {
            mvaddch(segment.y, segment.x, 'O');
        }
       
        mvaddch(snake.front().y, snake.front().x, '@');
        
        mvaddch(food.y, food.x, 'F');
        
        mvprintw(height, 0, "Score: %d | Steps: %d/%d", score, step_count, max_steps);
        refresh();
    }

    void init_ncurses() {
        initscr();
        if (!stdscr) {
            throw std::runtime_error("Failed to initialize ncurses");
        }
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        timeout(100);
        curs_set(0);
    }

    template<typename T>
    T get_distance(int dx, int dy) const {
        if (dx == 0 && dy == 0) return T(0);
        
        Position head = snake.front();
        int steps = 0;
        while (true) {
            head.x += dx;
            head.y += dy;
            steps++;
            
            if (head.x < 0 || head.x >= width || head.y < 0 || head.y >= height) {
                break;
            }
            
            bool body_found = false;
            for (const auto& segment : snake) {
                if (head == segment) {
                    body_found = true;
                    break;
                }
            }
            if (body_found) break;
        }
        
        return T(1) / T(steps);
    }

    void update_without_render() {
        step_count++;
        if (step_count > max_steps) {
            game_over = true;
            return;
        }
        
        Position head = snake.front();
        switch (direction) {
            case 0: head.y--; break;
            case 1: head.x++; break;
            case 2: head.y++; break;
            case 3: head.x--; break;
        }
        
        if (head.x <= 0 || head.x >= width-1 || head.y <= 0 || head.y >= height-1) {
            game_over = true;
            return;
        }
        
        for (auto it = snake.begin() + 1; it != snake.end(); ++it) {
            if (head == *it) {
                game_over = true;
                return;
            }
        }

        snake.push_front(head);

        if (head == food) {
            score += 10;
            place_food();
        } else {
            snake.pop_back();
        }
    }

public:
    SnakeGame(int w, int h, int max_steps = 10000) 
        : width(w), height(h), 
          food(0, 0), 
          direction(1), 
          game_over(false), 
          score(0),
          gen(std::random_device()()),
          dist_x(1, width-2),
          dist_y(1, height-2),
          step_count(0),
          max_steps(max_steps) {
        
        if (width < 5 || height < 5) {
            throw std::invalid_argument("Game area too small (minimum 5x5)");
        }

        int start_x = width / 2;
        int start_y = height / 2;
        snake.emplace_back(start_x, start_y);
        snake.emplace_back(start_x - 1, start_y);
        snake.emplace_back(start_x - 2, start_y);
        snake.emplace_back(start_x - 3, start_y);
        place_food();
    }

    int returnScore() const {
        return score;
    }

    Position returnFoodPlace() const {
        return food;
    }

    void setMaxSteps(int steps) {
        max_steps = steps;
    }

    void update_direction(int action) {
        if (action == 1) {
            direction = (direction + 1) % 4;
        } else if (action == 2) {
            direction = (direction + 3) % 4;
        }
    }

    template<typename T>
    std::vector<T> get_state() const {
        std::vector<T> state(8);
        Position head = snake.front();

        int dx_current, dy_current;
        switch (direction) {
            case 0: dx_current = 0; dy_current = -1; break;
            case 1: dx_current = 1; dy_current = 0; break;
            case 2: dx_current = 0; dy_current = 1; break;
            case 3: dx_current = -1; dy_current = 0; break;
        }

        state[0] = get_distance<T>(dx_current, dy_current);
        state[1] = get_distance<T>(dy_current, -dx_current);
        state[2] = get_distance<T>(-dy_current, dx_current);

        state[3] = static_cast<T>(food.x - head.x) / (width - 2);
        state[4] = static_cast<T>(food.y - head.y) / (height - 2);
        
        state[5] = static_cast<T>(dx_current);
        state[6] = static_cast<T>(dy_current);
        
        state[7] = static_cast<T>(snake.size() - 3) / ((width-2)*(height-2) - 3);

        return state;
    }

    template<typename T>
    int runWithoutRender(Perceptrone<T>& model) {
        while (!game_over) {
            auto state = get_state<T>();
            auto output = model.predict(state);
            
            int action = 0;
            T max_val = output[0];
            for (int i = 1; i < output.size(); i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    action = i;
                }
            }
            
            update_direction(action);
            update_without_render();
        }
        
        return score;
    }

    template<typename T>
    int runWithRender(Perceptrone<T>& model) {
        init_ncurses();
        
        while (!game_over) {
            auto state = get_state<T>();
            auto output = model.predict(state);
            
            int action = 0;
            T max_val = output[0];
            for (int i = 1; i < output.size(); i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    action = i;
                }
            }
            
            update_direction(action);
            update();
            draw();
            napms(50); 
        }
        
        endwin();
        return score;
    }

    void run() {
        try {
            init_ncurses();
            
            while (!game_over) {
                process_input(getch());
                update();
                draw();
            }
            
            endwin();
        } catch (const std::exception& e) {
            endwin();
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
};