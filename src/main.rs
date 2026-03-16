use ndarray::{Array1, Array2, array};
use rand::Rng;
use serde::{Deserialize, Serialize};
use core::net;
use std::fs::File;
use std::io::{self, BufRead, Write};


#[derive(Serialize, Deserialize, Clone)]
struct NeuralNet {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl NeuralNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-1.0..1.0));
        let b1 = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-1.0..1.0));
        let w2 = Array2::from_shape_fn((output_size, hidden_size), |_| rng.gen_range(-1.0..1.0));
        let b2 = Array1::from_shape_fn(output_size, |_| rng.gen_range(-1.0..1.0));
        NeuralNet { w1, b1, w2, b2 }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_deriv(y: f64) -> f64 {
        y * (1.0 - y)
    }

    fn forward(&self, inputs: &Array1<f64>) -> f64 {
        let z1 = self.w1.dot(inputs) + &self.b1;
        let a1 = z1.mapv(Self::sigmoid);
        let z2 = self.w2.dot(&a1) + &self.b2;
        Self::sigmoid(z2[0])
    }
fn train(&mut self, inputs_list: &Vec<Array1<f64>>, labels: &Vec<f64>, lr: f64, epochs: usize) {
    use ndarray::prelude::*;  // Axis, insert_axis, view(), to_owned() и т.д.

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (inputs, &y_true) in inputs_list.iter().zip(labels.iter()) {
            // ────────────────────────────────────────────────
            // Forward pass
            // ────────────────────────────────────────────────
            let z1 = self.w1.dot(inputs) + &self.b1;
            let a1 = z1.mapv(Self::sigmoid);           // [hidden_size]
            let z2 = self.w2.dot(&a1) + &self.b2;      // [1]
            let y_pred = Self::sigmoid(z2[0]);         // f64

            total_loss += (y_true - y_pred).powi(2);

            // ────────────────────────────────────────────────
            // Backward pass
            // ────────────────────────────────────────────────
            let error = y_pred - y_true;
            let delta_out = error * Self::sigmoid_deriv(y_pred);  // f64

            let delta_hidden = self.w2.t().dot(&array![delta_out]) * a1.mapv(Self::sigmoid_deriv);
            // delta_hidden : [hidden_size]

            // ────────────────────────────────────────────────
            // Обновление выходного слоя (w2, b2)
            // ────────────────────────────────────────────────
            // grad_w2 = delta_out × a1^T → (1, hidden_size)
            let a1_owned = a1.to_owned();                             // копия → owned
            let grad_w2 = delta_out * a1_owned.insert_axis(Axis(0));  // теперь умножение работает

            self.w2 -= &(grad_w2 * lr);
            self.b2 -= &array![delta_out * lr];

            // ────────────────────────────────────────────────
            // Обновление скрытого слоя (w1, b1)
            // ────────────────────────────────────────────────
            let delta_hidden_owned = delta_hidden.to_owned();          // копия → owned
            let inputs_owned = inputs.to_owned();                      // копия → owned

            let grad_w1 = delta_hidden_owned.clone()
                .insert_axis(Axis(1))                                  // (hidden_size, 1)
                .dot(&inputs_owned.insert_axis(Axis(0)));              // (1, input_size) → (hidden_size, input_size)

            self.w1 -= &(grad_w1 * lr);
            self.b1 -= &(delta_hidden_owned * lr);
        }

        if epoch % 500 == 0 || epoch == epochs - 1 {
            println!(
                "Эпоха {:6} | Средняя MSE: {:.8}",
                epoch,
                total_loss / inputs_list.len() as f64
            );
        }
    }
}

    fn save(&self, path: &str) -> io::Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let json: String = reader.lines().collect::<Result<_, _>>()?;
        let net: NeuralNet = serde_json::from_str(&json)?;
        Ok(net)
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "predict" {
        if args.len() !=4 {
            println!("Использование: cargo run predict 0 1");
            return Ok(());
        }

        let x1: f64 = args[2].parse().unwrap_or(0.0);
        let x2: f64 = args[3].parse().unwrap_or(0.0);

        match NeuralNet::load("xor_model.json") {
            Ok(net) => {
            let input = array![x1, x2];
            let pred = net.forward(&input);
            println!("XOR{:.0}, {:.0})~ {:.4} -> {}", x1, x2, pred, if pred > 0.5 {1} else {0});
            }
            Err(_) => println!("Модель не найдена. Сначала обучите сеть."),
        }
        return Ok(());

    }
    println!("Обучение сети на XOR...");
    let mut net  = NeuralNet::new(2,4,1);

    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];

    let labels = vec![0.0, 1.0, 1.0, 0.0];
    net.train(&inputs, &labels, 0.3, 10000);


    println!("/nПосле обучения:");
    for (inp, &lb1) in inputs.iter().zip(labels.iter()) {
        let pred = net.forward(inp);
       println!("{:?} -> {:.4} (Ожидается {})", inp.to_vec(), pred, lb1);
    }

    net.save("xor.model.json")?;
    println!("Модель сохранена в xor.model.json");
    println!("Теперь можно использовать: cargo run predict 0 1");

    Ok(())
}
 