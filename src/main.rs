use ndarray::{Array1, Array2, array};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_deriv(y: f64) -> f64 {
    y * (1.0 -y) 
}

fn mse_loss(y_true: f64, y_pred: f64) -> f64 {
    (y_true - y_pred).powi(2)
}

struct SimpleNet {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl SimpleNet {
    fn new() -> Self {
        SimpleNet {
            w1: array![[10.0, 10.0], [-10.0, -10.0]],
            b1: array![-15.0, 5.0],
            w2: array![[1.0, -1.0]],
            b2: array![0.0],
        }
    }

    fn forward(&self, inputs: &Array1<f64>) -> f64 {
        let z1 = self.w1.dot(inputs) + &self.b1;
        let a1 = z1.mapv(sigmoid);
        let z2 = self.w2.dot(&a1) + &self.b2;
        sigmoid(z2[0])
    }

    fn train_step(&mut self, inputs: &Array1<f64>, y_true: f64, lr: f64) {
        let y_pred = self.forward(inputs);
        let error = y_pred - y_true;
        let delta = error*sigmoid_deriv(y_pred);

        let z1 = self.w1.dot(inputs)+&self.b1;
        let a1 = z1.mapv(sigmoid);

        let w2_grad = &a1*delta;
        self.w2 = &self.w2 - &(w2_grad*lr).into_shape((1,2)).unwrap();
        self.b2 = &self.b2 - &(array![delta*lr]);
    }
}

fn main() {
    let mut net = SimpleNet::new();
    let labels = vec![0.0, 1.0, 1.0, 0.0];
    let inputs_list = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];

    let lr = 0.5;
    let epochs = 200; 

    for epochs in 0..epochs {
        let mut total_loss = 0.0;

        for (inputs, &y_true) in inputs_list.iter().zip(labels.iter()) {
            let y_pred = net.forward(inputs);
            total_loss += mse_loss(y_true, y_pred);

            net.train_step(inputs, y_true, lr);
        }

        if epochs % 200 ==0 {
            println!("Эпоха {} | Средняя ошибка: {:.6}", epochs, total_loss / 4.0);
        }
    }


   println!("/nПосле обучения:");
   for (inputs, &y_true) in inputs_list.iter().zip(labels.iter()) {
    let pred = net.forward(inputs);
    println!("Выход: {:?} -> Пред: {:.4} (правильно: {})", inputs.to_vec(), pred, y_true);
   }

}
