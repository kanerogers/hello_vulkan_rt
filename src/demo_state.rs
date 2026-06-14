pub const TRACK_LENGTH_METRES: f32 = 5_000.0;
pub const FAST_SPEED_MPS: f32 = 150.0 / 3.6; // 150kph
pub const SLOW_SPEED_MPS: f32 = 80.0 / 3.6; // 80kph
pub const ACCEL_RESPONSE_HZ: f32 = 0.35;
pub const CURVE_RADIUS_METRES: f32 = 250.0;
pub const STRAIGHT_LENGTH_METRES: f32 = 1_200.0;

#[derive(Debug)]
pub struct DemoState {
    pub elapsed_s: f32,
    pub track_s_m: f32,
    pub speed_mps: f32,
    pub target_speed_mps: f32,
    pub next_debug_log_s: f32,
    pub debug_frame_count: u32,
    pub debug_frame_time_s: f32,
}

impl DemoState {
    pub fn record_frame(&mut self, frame_time_s: f32) {
        self.debug_frame_count += 1;
        self.debug_frame_time_s += frame_time_s;
    }

    pub fn update(&mut self, dt: f32) {
        self.elapsed_s += dt;
        self.target_speed_mps = self.target_speed();

        let response = 1.0 - (-ACCEL_RESPONSE_HZ * dt).exp();
        self.speed_mps += (self.target_speed_mps - self.speed_mps) * response;

        self.track_s_m = (self.track_s_m + self.speed_mps * dt).rem_euclid(TRACK_LENGTH_METRES);

        if self.elapsed_s >= self.next_debug_log_s {
            self.next_debug_log_s += 1.0;

            let fps = if self.debug_frame_time_s > 0.0 {
                self.debug_frame_count as f32 / self.debug_frame_time_s
            } else {
                0.0
            };

            log::info!(
                "choOOoOO: elapsed={:.1}s, s={:.1}. speed={:.1}km/h target={:.1}km/h fps={:.1}",
                self.elapsed_s,
                self.track_s_m,
                self.speed_mps * 3.6,
                self.target_speed_mps * 3.6,
                fps,
            );

            self.debug_frame_count = 0;
            self.debug_frame_time_s = 0.0;
        }
    }

    fn target_speed(&self) -> f32 {
        if (250.0..900.0).contains(&self.track_s_m) {
            FAST_SPEED_MPS
        } else {
            SLOW_SPEED_MPS
        }
    }
}

impl Default for DemoState {
    fn default() -> Self {
        Self {
            elapsed_s: Default::default(),
            track_s_m: Default::default(),
            speed_mps: Default::default(),
            target_speed_mps: FAST_SPEED_MPS,
            next_debug_log_s: Default::default(),
            debug_frame_count: Default::default(),
            debug_frame_time_s: Default::default(),
        }
    }
}
