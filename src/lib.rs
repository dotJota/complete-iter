use std::collections::HashMap;

pub mod models;
pub mod helper;

pub struct Agent {
    system_state: models::SystemState,
    policy: HashMap<i64,HashMap<String,f64>>,
    policy_evaluation: HashMap<i64,f64>,
}

impl Agent {

    pub fn init_random(system_state: models::SystemState) -> Agent {

        let policy: HashMap<i64,HashMap<String,f64>> = system_state
            .get_all_states()
            .iter()
            .map(|(id, state)| (*id, state.get_random_policy()))
            .collect();

        let policy_evaluation: HashMap<i64,f64> = system_state.get_all_states()
            .iter().map(|(id, _)| (*id, 0.)).collect();

        return Agent {system_state, policy, policy_evaluation}
    }

    pub fn set_polity(&mut self, policy: HashMap<i64,HashMap<String,f64>>) {
        self.policy = policy;
    }

    pub fn get_policy(&self) -> &HashMap<i64,HashMap<String,f64>> {
        return &self.policy
    }

    pub fn get_best_action(&self, state_id: i64) -> Option<(&String,&f64)> {
        self.policy.get(&state_id).unwrap().iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    }

    pub fn get_evaluation(&self) -> &HashMap<i64,f64> {
        return &self.policy_evaluation
    }

    pub fn get_system_state(&self) -> &models::SystemState {
        return &self.system_state
    }

    pub fn evaluate_policy(&mut self, gamma: f64, epsilon: f64, n_iter: u32) {

        // rewards
        // policy: HashMap<i64,HashMap<String,f64>>
        let static_rewards: HashMap<i64,f64> = self.policy
            .iter().map(|(id, actions_prob)| {
                let actions_reward = self.system_state.get_state(id).unwrap().get_eval_rewards();
                (*id, helper::match_mul_sum(actions_prob, actions_reward))
            }).collect();

        // transition_probs: HashMap<String,HashMap<i64,f64>>
        let state_probs: HashMap<i64,HashMap<i64,f64>> = self.policy
            .iter().map(|(id_prev, action_prob)| {
                let transition_probs: HashMap<i64,f64> = self.system_state.get_state(id_prev)
                    .unwrap().get_eval_probs()
                    .iter().map(|(id_next, transition_prob)| {
                        (*id_next, helper::match_mul_sum(action_prob, transition_prob))
                    }).collect();
                (*id_prev, transition_probs)
            }).collect();

        // Iterative policy evaluation
        let mut counter: u32 = 0;

        loop {
            let mut delta = 0.;

            self.policy_evaluation = self.policy_evaluation.iter()
            .map(|(id, value)| {
                let future_reward = gamma*helper::match_mul_sum(state_probs.get(id).unwrap(), &self.policy_evaluation);
                let new_reward = static_rewards.get(id).unwrap() + future_reward;
                delta = f64::max(delta, (new_reward - value).abs());
                (*id, new_reward)
            }).collect();

            counter += 1;

            if (delta < epsilon) || (counter == n_iter) {
                break
            }
        }
        
    }

    pub fn deterministic_policy_improvement(&mut self, gamma: f64, epsilon: f64, policy_iters: u32, eval_iters: u32) {
        
        // Default string for states with no actions
        let default_str = "_No_Actions_".to_string();
        self.evaluate_policy(gamma, epsilon, eval_iters);

        let mut policy_counter: u32 = 0;

        loop {
            let old_eval = self.policy_evaluation.clone();

            self.policy = self.system_state.get_all_states().iter()
                .map(|(id, state)| {
                    let best_action = self.calc_best_action(state, &default_str);
                    (*id, self.calc_best_policy(state, best_action))
                }).collect();

            self.evaluate_policy(gamma, epsilon, 100);

            let max_diff: f64 = old_eval.iter()
            .map(|(id, old_val)| {
                let new_val = self.policy_evaluation.get(id).unwrap();
                (old_val - new_val).abs()
            }).max_by(|a,b| a.partial_cmp(b).unwrap())
            .unwrap();
            
            policy_counter += 1;
            if (max_diff < epsilon) || (policy_counter == policy_iters) {
                break;
            }

        }

    }

    pub fn calc_best_action<'a>(&'a self, state: &'a models::ModelState, default_str: &'a String) -> &'a String {

        let max_action_reward: &String = state.get_all_probs().iter()
            .map(|(action, probs)| {
                let action_reward = state.get_eval_rewards().get(action).unwrap();
                let future_reward = helper::match_mul_sum(probs, &self.policy_evaluation);
                (action, action_reward + future_reward)
            })
            .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((default_str, 0.)).0;

        return max_action_reward

    }

    pub fn calc_best_policy(&self, state: &models::ModelState, best_action: &String) -> HashMap<String,f64> {
        return state.get_eval_rewards().iter()
            .map(|(action, _)| {
                if *action == *best_action {
                    (action.clone(), 1.)
                } else {
                    (action.clone(), 0.)
                }
            }).collect()
    }

}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn policy_initialization_test() {
        let action_1 = String::from("First_Action");
        let action_2 = String::from("Second_Action");
        let action_3 = String::from("Third_action");

        let links = vec![
            models::StateLink(0, 1, action_1.clone(), 0.9, 0.),
            models::StateLink(0, 2, action_1.clone(), 0.1, 10.),
            models::StateLink(0, 0, action_2.clone(), 0.9, 0.),
            models::StateLink(0, 1, action_2.clone(), 0.1, 0.),
            models::StateLink(0, 1, action_3.clone(), 1., 1.),
            models::StateLink(1, 2, action_1.clone(), 1., 5.),
            models::StateLink(1, 0, action_2.clone(), 0.5, 0.),
            models::StateLink(1, 2, action_2.clone(), 0.5, 0.),
        ];

        let test_system = models::SystemState::create_and_build(links);

        let test_agent = Agent::init_random(test_system);

        let mut random_policy: HashMap<i64, HashMap<String,f64>> = HashMap::new();

        let mut policy_0: HashMap<String,f64> = HashMap::new();
        policy_0.insert(action_1.clone(), 1./3.);
        policy_0.insert(action_2.clone(), 1./3.);
        policy_0.insert(action_3.clone(), 1./3.);

        let mut policy_1: HashMap<String,f64> = HashMap::new();
        policy_1.insert(action_1.clone(), 1./2.);
        policy_1.insert(action_2.clone(), 1./2.);

        random_policy.insert(0, policy_0);
        random_policy.insert(1, policy_1);
        random_policy.insert(2, HashMap::new());

        assert_eq!(*test_agent.get_policy(), random_policy);

    }

    #[test]
    fn policy_eval_test_1() {
        // Simple n-armed model with a single attempt
        let arms = ["Arm_1".to_string(), "Arm_2".to_string(), "Arm_3".to_string()];
        let links = vec![
            models::StateLink(0, 1, arms[0].clone(), 1., 1.),
            models::StateLink(0, 1, arms[1].clone(), 1., 2.),
            models::StateLink(0, 1, arms[2].clone(), 1., 3.),
        ];

        let system_state = models::SystemState::create_and_build(links);
        let mut test_agent = Agent::init_random(system_state);

        let epsilon = 0.01;
        test_agent.evaluate_policy(1., epsilon, 10);

        let expected_evaluation = 2.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

        let mut new_policy: HashMap<i64,HashMap<String,f64>> = HashMap::new();

        let mut policy_0: HashMap<String,f64> = HashMap::new();
        policy_0.insert("Arm_1".to_string(), 0.);
        policy_0.insert("Arm_2".to_string(), 0.);
        policy_0.insert("Arm_3".to_string(), 1.);

        new_policy.insert(0, policy_0);
        new_policy.insert(1, HashMap::new());

        test_agent.set_polity(new_policy);
        test_agent.evaluate_policy(1., epsilon, 10);

        let expected_evaluation = 3.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);
    }

    #[test]
    fn policy_eval_test_2() {
        // Two n-armed model with a single attempt each
        let arms = ["Arm_1".to_string(), "Arm_2".to_string(), "Arm_3".to_string()];
        let links = vec![
            models::StateLink(0, 1, arms[0].clone(), 1., 1.),
            models::StateLink(0, 1, arms[1].clone(), 1., 2.),
            models::StateLink(0, 1, arms[2].clone(), 1., 3.),
            models::StateLink(1, 2, arms[0].clone(), 1., 3.),
            models::StateLink(1, 2, arms[1].clone(), 1., 2.),
            models::StateLink(1, 2, arms[2].clone(), 1., 1.),
        ];

        let system_state = models::SystemState::create_and_build(links);
        let mut test_agent = Agent::init_random(system_state);

        let epsilon = 0.01;
        test_agent.evaluate_policy(1., epsilon, 10);

        let expected_evaluation = 4.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

        let expected_evaluation = 2.;
        let diff = (test_agent.get_evaluation().get(&1).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

        let mut new_policy: HashMap<i64,HashMap<String,f64>> = HashMap::new();

        let mut policy_0: HashMap<String,f64> = HashMap::new();
        policy_0.insert("Arm_1".to_string(), 0.);
        policy_0.insert("Arm_2".to_string(), 0.);
        policy_0.insert("Arm_3".to_string(), 1.);

        let mut policy_1: HashMap<String,f64> = HashMap::new();
        policy_1.insert("Arm_1".to_string(), 1.);
        policy_1.insert("Arm_2".to_string(), 0.);
        policy_1.insert("Arm_3".to_string(), 0.);

        new_policy.insert(0, policy_0);
        new_policy.insert(1, policy_1);
        new_policy.insert(2, HashMap::new());

        test_agent.set_polity(new_policy);
        test_agent.evaluate_policy(1., epsilon, 10);

        let expected_evaluation = 6.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

        let expected_evaluation = 3.;
        let diff = (test_agent.get_evaluation().get(&1).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

    }

    #[test]
    pub fn policy_improv_test_1() {
        // Simple n-armed model with a single attempt
        let arms = ["Arm_1".to_string(), "Arm_2".to_string(), "Arm_3".to_string()];
        let links = vec![
            models::StateLink(0, 1, arms[0].clone(), 1., 1.),
            models::StateLink(0, 1, arms[1].clone(), 1., 2.),
            models::StateLink(0, 1, arms[2].clone(), 1., 3.),
        ];

        let system_state = models::SystemState::create_and_build(links);
        let mut test_agent = Agent::init_random(system_state);

        let epsilon = 0.01;
        test_agent.deterministic_policy_improvement(1., epsilon, 100, 100);

        let expected_evaluation = 3.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        // Prints only when it fails
        println!("Policy: {:?}", test_agent.get_policy());
        println!("Eval 0: {:?}", test_agent.get_evaluation().get(&0).unwrap());

        assert!(diff < 2.*epsilon);
    }

    #[test]
    pub fn policy_improv_test_2() {
        // Two n-armed model with a single attempt each
        let arms = ["Arm_1".to_string(), "Arm_2".to_string(), "Arm_3".to_string()];
        let links = vec![
            models::StateLink(0, 1, arms[0].clone(), 1., 1.),
            models::StateLink(0, 1, arms[1].clone(), 1., 2.),
            models::StateLink(0, 1, arms[2].clone(), 1., 3.),
            models::StateLink(1, 2, arms[0].clone(), 1., 3.),
            models::StateLink(1, 2, arms[1].clone(), 1., 2.),
            models::StateLink(1, 2, arms[2].clone(), 1., 1.),
        ];

        let system_state = models::SystemState::create_and_build(links);
        let mut test_agent = Agent::init_random(system_state);

        let epsilon = 0.01;
        test_agent.deterministic_policy_improvement(1., epsilon, 100, 100);

        // Prints only when it fails
        println!("Policy: {:?}", test_agent.get_policy());
        println!("Eval 0: {:?}", test_agent.get_evaluation().get(&0).unwrap());
        println!("Eval 1: {:?}", test_agent.get_evaluation().get(&1).unwrap());

        let expected_evaluation = 6.;
        let diff = (test_agent.get_evaluation().get(&0).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);

        let expected_evaluation = 3.;
        let diff = (test_agent.get_evaluation().get(&1).unwrap() - expected_evaluation).abs();

        assert!(diff < 2.*epsilon);
    }

}
