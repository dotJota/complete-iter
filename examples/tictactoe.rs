/* 
    Game TicTacToe
*/

use std::io;

use complete_iter::{models, Agent};

#[derive(Copy, Clone)]
enum  Mark {
    Cross,
    Circle,
    Empty
}

impl Mark {

    fn flip(&self) -> Mark {
        match self {
            Mark::Cross => Mark::Circle,
            Mark::Circle => Mark::Cross,
            Mark::Empty => Mark::Empty
        }
    }
    
    fn is_equal(&self, other: Mark) -> bool {
        match self {
            Mark::Cross => {
                match other {
                    Mark::Cross => true,
                    _ => false
                }
            },
            Mark::Circle => {
                match other {
                    Mark::Circle => true,
                    _ => false
                }
            },
            Mark::Empty => {
                match other {
                    Mark::Empty => true,
                    _ => false
                }
            }
        }
    }
    
    fn to_string(&self) -> String {
        match self {
            Mark::Cross => "X".to_string(),
            Mark::Circle => "O".to_string(),
            Mark::Empty => " ".to_string()
        }
    }

    // Each mark has an associated id:
    // Empty = 0, Cicle = 1, Cross = 2
    fn get_id(&self) -> i64 {
        match self {
            Mark::Empty => 0,
            Mark::Circle => 1,
            Mark::Cross => 2
        }
    }

    fn from_id(id: i64) -> Mark {
        match id {
            1 => Mark::Circle,
            2 => Mark::Cross,
            _ => Mark::Empty
        }
    }
    
}

struct TicTacBoard{
    board: [[Mark; 3]; 3],
    actions: [String; 9]
}

impl TicTacBoard {

    fn new() -> TicTacBoard {
        let actions = [
            String::from("[0,0]"),
            String::from("[0,1]"),
            String::from("[0,2]"),
            String::from("[1,0]"),
            String::from("[1,1]"),
            String::from("[1,2]"),
            String::from("[2,0]"),
            String::from("[2,1]"),
            String::from("[2,2]")
        ];
        return TicTacBoard {board: [[Mark::Empty; 3]; 3], actions};
    }

    // Returns a game with board matching id and player's turn
    fn from_id(id: i64) -> (TicTacBoard, Mark) {
        assert!((id >= 0) & (id < 3_i64.pow(9_u32) - 1));
        let mut game = TicTacBoard::new();
        let mut remaining = id;
        let mut n_circle = 0;
        let mut n_cross = 0;

        for action in game.actions.clone() {
            let cell_id = remaining % 3;

            match cell_id {
                1 => n_circle += 1,
                2 => n_cross += 1,
                _ => ()
            };

            remaining = remaining / 3;
            game.apply_action(&action, Mark::from_id(cell_id));
        }

        let player;
        if n_circle > n_cross {
            player = Mark::Cross;
        } else {
            player = Mark::Circle;
        }

        return (game, player)

    }

    pub fn get_state_id(&self) -> i64 {
        let mut counter: i64 = -1;
        let mut id = 0;

        for row in self.board {
            for cell in row {
                counter += 1;
                id += 3_i64.pow(counter as u32)*cell.get_id();
            }
        }

        return id
    }

    pub fn possible_actions(&self) -> Vec<String> {
        let mut act_iter = self.actions.iter();
        let mut output: Vec<String> = Vec::new();

        for row in self.board {
            for cell in row {
                let action = act_iter.next().unwrap();
                if cell.is_equal(Mark::Empty) {
                    output.push(action.clone());
                }
            }
        }

        return output
    }

    pub fn apply_action(&mut self, action: &String, player: Mark) {
        let row: usize = action[1..2].parse().unwrap();
        let col: usize = action[3..4].parse().unwrap();
        self.board[row][col] = player;
    }

    pub fn roll_back(&mut self, action: &String) {
        let row: usize = action[1..2].parse().unwrap();
        let col: usize = action[3..4].parse().unwrap();
        self.board[row][col] = Mark::Empty;
    }

    pub fn has_won(&self, player: Mark) -> bool {

        // Check rows
        for i in 0..3 {
            let win_con = self.board[i][0].is_equal(player)
                        && self.board[i][1].is_equal(player)
                        && self.board[i][2].is_equal(player);
            if win_con {
                return true
            }
        }
        
        // Check columns
        for j in 0..3 {
            let win_con = self.board[0][j].is_equal(player)
                        && self.board[1][j].is_equal(player)
                        && self.board[2][j].is_equal(player);
            if win_con {
                return true
            }
        }
        
        // Check diagonals
        let win_con_1 = self.board[0][0].is_equal(player)
                        && self.board[1][1].is_equal(player)
                        && self.board[2][2].is_equal(player);
        
        let win_con_2 = self.board[0][2].is_equal(player)
                        && self.board[1][1].is_equal(player)
                        && self.board[2][0].is_equal(player);
        
        return win_con_1 || win_con_2

    }

    // Prints the current state of the game
    pub fn to_string(&self) {
        let mut print_str = String::new();
        for i in 0..3 {
            for j in 0..3 {
                print_str.push_str(&self.board[i][j].to_string());
                
                if j < 2 {
                    print_str.push_str(" | ");
                }
            }
            
            if i < 2 {
                print_str.push_str("\n---------\n");
            }
        }
        println!("\n{}\n", print_str);
    }

}

fn main() {

    let mut ids_seen: Vec<i64> = Vec::new();
    let mut ids_done: Vec<i64> = Vec::new();
    let mut links: Vec<models::StateLink> = Vec::new();

    ids_seen.push(0);

    loop {
        let id = match ids_seen.pop() {
            Some(number) => number,
            None => break
        };

        if ids_done.contains(&id) {
            continue;
        }

        let (mut game, player) = TicTacBoard::from_id(id);
        add_links(&mut game, &mut links, player, &mut ids_seen);

        ids_done.push(id);
    }

    let tic_tac_state = models::SystemState::create_and_build(links);
    let mut tic_tac_agent = Agent::init_random(tic_tac_state);
    tic_tac_agent.deterministic_policy_improvement(1., 0.01, 100, 100);

    /*
    // Let's see the AI play

    let mut game = TicTacBoard::new();
    let mut player = Mark::Circle;

    game.to_string();

    loop {

        let next_action = match tic_tac_agent.get_best_action(game.get_state_id()) {
            Some((action, _)) => action,
            None => break,
        };

        game.apply_action(next_action, player);
        
        player = player.flip();

        game.to_string();

        println!("X has won: {}", game.has_won(Mark::Cross));
        println!("O has won: {}\n", game.has_won(Mark::Circle));

    }
    */

    play_with_agent(&tic_tac_agent);

}

// Considers a random adversary policy
fn add_links(game: &mut TicTacBoard, links: &mut Vec<models::StateLink>, player: Mark, ids_seen: &mut Vec<i64>) {

    if game.has_won(player) || game.has_won(player.flip()) {
        return;
    }
    
    let id_prev = game.get_state_id();
    let actions = game.possible_actions();

    for action_player in &actions {

        game.apply_action(action_player, player);
        let id_next = game.get_state_id();
        ids_seen.push(id_next);

        if game.has_won(player) {
            links.push(models::StateLink(id_prev, id_next, action_player.to_string(), 1., 1.));

        } else {

            for action_adv in &actions {
                if *action_player != *action_adv {

                    game.apply_action(action_adv, player.flip());
                    let id_next = game.get_state_id();
                    let reward = if game.has_won(player.flip()) {-1.} else {0.};
                    let prob = 1./(actions.len() - 1) as f64;

                    links.push(models::StateLink(id_prev, id_next, action_player.to_string(), prob, reward));

                    ids_seen.push(id_next);
                    game.roll_back(action_adv);
                }
            }

        }

        game.roll_back(action_player);

    }

}


pub fn play_with_agent(tic_tac_agent: &Agent) {

    loop {
        
        let mut game = TicTacBoard::new();
        let bot = Mark::Circle;
        let human = Mark::Cross;

        println!("A new game started against the bot.");

        game.to_string();

        loop {

            let next_action = match tic_tac_agent.get_best_action(game.get_state_id()) {
                Some((action, _)) => action,
                None => break,
            };

            println!("The bot played at {}", next_action);

            game.apply_action(next_action, bot);

            game.to_string();

            if game.has_won(bot) {
                println!("\nThe bot won!!");
                break;
            }

            loop {
                
                let possible_actions = game.possible_actions();

                println!("\nYour turn, please type one of the following actions: \n{:?}", &possible_actions);

                let mut play = String::new();

                match io::stdin().read_line(&mut play) {
                    Ok(_) => (),
                    Err(_) => {
                        println!("Incorrect input, please try again.");
                        continue
                    },
                };

                if possible_actions.contains(&play.trim().to_string()) {

                    println!("You played at {}", &play);

                    game.apply_action(&play, human);

                    game.to_string();

                    break;

                } else {
                    println!("Please try again with a valid play.");
                }
            
            }

            if game.has_won(human) {
                println!("\nYou won!!");
                break;
            }

        }

        println!("Eng of game! Wanna play again? (y/n)");

        let mut answer = String::new();

        match io::stdin().read_line(&mut answer) {
            Ok(_) => (),
            Err(_) => {
                println!("Incorrect input, please try again.");
                continue
            },
        };

        if answer.trim() == "y" {
            continue;
        } else {
            break;
        }

    }
    

}
