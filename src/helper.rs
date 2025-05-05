use std::{collections::HashMap, hash::Hash};

// A function that computes the product of items with matching keys
pub fn match_mul<'a,T: Eq + Hash>(map_1: &'a HashMap<T,f64>, map_2: &'a HashMap<T,f64>) -> HashMap<&'a T,f64> {
    return map_1.iter()
        .map(|(key, value)| (key, map_2.get(key).unwrap_or(&0.)*value))
        .collect()
}

// Computes the sum of the product of items with matching keys
pub fn match_mul_sum<T: Eq + Hash>(map_1: &HashMap<T,f64>, map_2: &HashMap<T,f64>) -> f64 {
    return map_1.iter()
        .map(|(key, value)| map_2.get(key).unwrap_or(&0.)*value)
        .sum()
}

#[cfg(test)]
mod tests {

    use super::*;

    // Test HashMap multiplication function
    #[test]
    fn hash_multiply_test() {
        let mut map_1: HashMap<String,f64> = HashMap::new();
        let mut map_2: HashMap<String,f64> = HashMap::new();
        let mut map_3: HashMap<&String,f64> = HashMap::new();
        let keys = vec!["Key_1".to_string(), "Key_2".to_string(), "Key_3".to_string()];

        for key in &keys{
            map_1.insert(key.clone(), 0.5);
            map_2.insert(key.clone(), 2.);
            map_3.insert(key, 1.);
        }

        let key_4 = "Key_4".to_string();
        map_1.insert(key_4.clone(), 10.);
        map_3.insert(&key_4, 0.);

        assert_eq!(match_mul(&map_1, &map_2), map_3);
    }

    // Test multiply reduce function
    #[test]
    fn hash_multiply_reduce_test() {
        let mut map_1: HashMap<String,f64> = HashMap::new();
        let mut map_2: HashMap<String,f64> = HashMap::new();

        let keys = vec!["Key_1".to_string(), "Key_2".to_string(), "Key_3".to_string()];

        for key in &keys{
            if key != "Key_3" {
                map_1.insert(key.clone(), 0.5);
            }
            map_2.insert(key.clone(), 2.);
        }

        map_1.insert("Key_4".to_string(), 10.);

        assert_eq!(match_mul_sum(&map_1, &map_2), 2.);

    }

}