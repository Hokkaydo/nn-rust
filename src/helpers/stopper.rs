pub struct PlateauDetector {
    best_loss: f32,
    patience: usize,
    wait: usize,
    min_delta: f32,
}

impl PlateauDetector {
    pub(crate) fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            best_loss: f32::INFINITY,
            patience,
            wait: 0,
            min_delta,
        }
    }

    pub fn has_plateaued(&mut self, current_loss: f32) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
        }
        self.wait >= self.patience
    }
}
