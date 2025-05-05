use std::collections::HashMap;

// Model states
#[derive(Debug, PartialEq)]
pub struct ModelState {
    state_id: i64,
    transition_probs: HashMap<String,HashMap<i64,f64>>,
    action_rewards: HashMap<String,HashMap<i64,f64>>,
    state_reward: f64,
    eval_action_rewards: HashMap<String,f64>,
    eval_transition_probs: HashMap<i64,HashMap<String,f64>>
}

impl ModelState {

    pub fn new(id: i64) -> ModelState {
        let mut state = ModelState {
            state_id: id,
            transition_probs: HashMap::new(),
            action_rewards: HashMap::new(),
            state_reward: 0.,
            eval_action_rewards: HashMap::new(),
            eval_transition_probs: HashMap::new()
        };

        state.calc_eval_rewards();

        return state
    }

    pub fn insert_link(&mut self, new_state: i64, action: &String, prob: f64, reward: f64) {
        self.transition_probs.entry(action.clone())
            .or_insert(HashMap::new())
            .insert(new_state, prob);

        self.action_rewards.entry(action.clone())
            .or_insert(HashMap::new())
            .insert(new_state, reward);
    }

    pub fn set_reward(&mut self, new_reward: f64) {
        self.state_reward = new_reward;
    }
    
    pub fn get_id(&self) -> i64 {
        return self.state_id
    }

    pub fn get_all_probs(&self) -> &HashMap<String,HashMap<i64,f64>> {
        return &self.transition_probs
    }

    pub fn get_probs(&self, action: &String) -> Option<&HashMap<i64,f64>> {
        return self.transition_probs.get(action)
    }

    pub fn get_all_action_rewards(&self) -> &HashMap<String,HashMap<i64,f64>> {
        return &self.action_rewards
    }

    pub fn get_action_reward(&self, action: &String) -> Option<&HashMap<i64,f64>> {
        return self.action_rewards.get(action)
    }

    pub fn get_reward(&self) -> f64 {
        return self.state_reward
    }

    // Support functions for Actor

    pub fn get_random_policy(&self) -> HashMap<String,f64> {
        self.action_rewards
            .iter()
            .map(|(action, _)| (action.clone(), 1./self.action_rewards.len() as f64))
            .collect()
    }

    pub fn calc_eval_rewards(&mut self) {
        self.eval_action_rewards = self.action_rewards.iter()
        .map(|(action, rewards)| {
            (
                action.clone(),
                rewards.iter().map(|(id,reward)| {
                    self.get_probs(action).unwrap().get(id).unwrap()*reward
                }).sum()
            )
        }).collect();
    }

    pub fn calc_eval_transition(&mut self) {
        let mut new_eval_transition: HashMap<i64,HashMap<String,f64>> = HashMap::new();

        for (action, probs) in &self.transition_probs {
            for (id, prob) in probs {
                new_eval_transition.entry(*id).or_insert(HashMap::new())
                    .insert(action.clone(), *prob);
            }
        }

        for (_, map) in &mut new_eval_transition {
            for (action, _) in &self.transition_probs {
                map.entry(action.clone()).or_insert(0.);
            }
        }

        self.eval_transition_probs = new_eval_transition;
    }

    pub fn get_eval_rewards(&self) -> &HashMap<String,f64> {
        return &self.eval_action_rewards
    }

    pub fn get_eval_probs(&self) -> &HashMap<i64,HashMap<String,f64>> {
        return &self.eval_transition_probs
    }

}

// Transition between states given an action
// (prev_state, new_state, action, probability, reward)
#[derive(Debug, PartialEq)]
pub struct StateLink(pub i64, pub i64, pub String, pub f64, pub f64);

#[derive(Debug, PartialEq)]
pub struct SystemState {
    states: HashMap<i64,ModelState>,
    speficication: Vec<StateLink>,
    is_built: bool,
}

impl SystemState {

    pub fn create_and_build(links: Vec<StateLink>) -> SystemState {
        let mut system_state = SystemState {
            states: HashMap::new(),
            speficication: links,
            is_built: false
        };

        system_state.build();

        return system_state
    }
    
    pub fn build(&mut self) {
        
        for link in &self.speficication {
            // (prev_state, new_state, action, probability, reward)
            self.states.entry(link.0)
                .or_insert(ModelState::new(link.0))
                .insert_link(link.1, &link.2, link.3, link.4);

            self.states.entry(link.1).or_insert(ModelState::new(link.1));
        }

        for (_, state) in self.states.iter_mut() {
            state.calc_eval_rewards();
            state.calc_eval_transition();
        }

        self.is_built = true;
    }

    pub fn get_state(&self, id: &i64) -> Option<&ModelState> {
        return self.states.get(id)
    }

    pub fn get_all_states(&self) -> &HashMap<i64,ModelState> {
        return &self.states
    }

}


#[cfg(test)]
mod tests {

    use super::*;

    // Simple system state creation
    #[test]
    fn creation_test() {
        // A state with a single action that points to itself
        let action = String::from("Single_Action");
        let mut transition_probs: HashMap<String,HashMap<i64,f64>> = HashMap::new();
        transition_probs.insert(action.clone(), HashMap::new());
        transition_probs.get_mut(&action).unwrap().insert(0, 1.);

        let mut action_rewards: HashMap<String,HashMap<i64,f64>> = HashMap::new();
        action_rewards.insert(action.clone(), HashMap::new());
        action_rewards.get_mut(&action).unwrap().insert(0, 10.);

        let mut test_state = ModelState {
            state_id: 0,
            transition_probs,
            action_rewards,
            state_reward: 0.,
            eval_action_rewards: HashMap::new(),
            eval_transition_probs: HashMap::new()
        };

        test_state.calc_eval_rewards();
        test_state.calc_eval_transition();

        let links = vec![StateLink(0, 0, "Single_Action".to_string(), 1., 10.)];
        let mut test_system = SystemState{
            states: HashMap::new(),
            speficication: links,
            is_built: false,
        };

        test_system.build();

        assert_eq!(test_state,*test_system.get_state(&0).unwrap());

    }

    // Model with initial and final state
    #[test]
    fn transition_test() {
        // An initial state and an end state
        // Two actions, one leads to end without reward
        // Other leads to either same state or end with a reward
            
        let action_1 = String::from("First_Action");
        let action_2 = String::from("Second_Action");
        let mut transition_probs: HashMap<String,HashMap<i64,f64>> = HashMap::new();
        let mut action_rewards: HashMap<String,HashMap<i64,f64>> = HashMap::new();

        // First action transition and rewards
        transition_probs.insert(action_1.clone(), HashMap::new());
        transition_probs.get_mut(&action_1).unwrap().insert(1, 1.);

        action_rewards.insert(action_1.clone(), HashMap::new());
        action_rewards.get_mut(&action_1).unwrap().insert(1, 0.);

        // Second action transition and rewards
        transition_probs.insert(action_2.clone(), HashMap::new());
        transition_probs.get_mut(&action_2).unwrap().insert(0, 0.9);
        transition_probs.get_mut(&action_2).unwrap().insert(1, 0.1);

        action_rewards.insert(action_2.clone(), HashMap::new());
        action_rewards.get_mut(&action_2).unwrap().insert(0, 0.);
        action_rewards.get_mut(&action_2).unwrap().insert(1, 10.);

        let mut test_state_1 = ModelState {
            state_id: 0,
            transition_probs,
            action_rewards,
            state_reward: 0.,
            eval_action_rewards: HashMap::new(),
            eval_transition_probs: HashMap::new()
        };

        test_state_1.calc_eval_rewards();
        test_state_1.calc_eval_transition();

        let mut test_state_2 = ModelState {
            state_id: 1,
            transition_probs: HashMap::new(),
            action_rewards: HashMap::new(),
            state_reward: 0.,
            eval_action_rewards: HashMap::new(),
            eval_transition_probs: HashMap::new()
        };

        test_state_2.calc_eval_rewards();
        test_state_2.calc_eval_transition();

        let mut test_states: HashMap<i64, ModelState> = HashMap::new();
        test_states.insert(0, test_state_1);
        test_states.insert(1, test_state_2);

        // Using built in builder
        let links = vec![
            StateLink(0, 1, action_1.clone(), 1., 0.),
            StateLink(0, 0, action_2.clone(), 0.9, 0.),
            StateLink(0, 1, action_2.clone(), 0.1, 10.),
        ];

        let mut test_system = SystemState{
            states: HashMap::new(),
            speficication: links,
            is_built: false,
        };

        test_system.build();

        assert_eq!(test_states,*test_system.get_all_states());
    }

    // Test eval_action_rewards and eval_transition_probs
    #[test]
    fn eval_action_rewards_test() {
        // An initial state and an end state
        // Two actions, one leads to end without reward
        // Other leads to either same state or end with a reward

        let action_1 = String::from("First_Action");
        let action_2 = String::from("Second_Action");

        let links = vec![
            StateLink(0, 1, action_1.clone(), 1., 0.),
            StateLink(0, 0, action_2.clone(), 0.9, 0.),
            StateLink(0, 1, action_2.clone(), 0.1, 10.),
        ];

        let mut test_system = SystemState{
            states: HashMap::new(),
            speficication: links,
            is_built: false,
        };

        test_system.build();

        let expected_rewards: HashMap<String,f64> = [(action_1.clone(), 0.), (action_2.clone(), 1.)]
            .iter().map(|x| x.clone()).collect();

        let mut expected_probs: HashMap<i64,HashMap<String,f64>> = HashMap::new();
        let probs_0: HashMap<String,f64> = [(action_1.clone(), 0.), (action_2.clone(), 0.9)]
            .iter().map(|x| x.clone()).collect();
        let probs_1: HashMap<String,f64> = [(action_1.clone(), 1.), (action_2.clone(), 0.1)]
            .iter().map(|x| x.clone()).collect();

        expected_probs.insert(0, probs_0);
        expected_probs.insert(1, probs_1);

        assert_eq!(*test_system.get_state(&0).unwrap().get_eval_rewards(), expected_rewards);
        assert_eq!(*test_system.get_state(&0).unwrap().get_eval_probs(), expected_probs);

    }

}
