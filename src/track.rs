use crate::demo_state::{CURVE_RADIUS_METRES, STRAIGHT_LENGTH_METRES, TRACK_LENGTH_METRES};

/// A sampled coordinate frame on the track.
///
/// All values are in world space.
/// `s_m` is the wrapped distance along the track, in metres.
///
/// Convention:
/// - `forward` is the tangent direction of the rail/tunnel.
/// - `right` is lateral +x from the track centre.
/// - `up` is world +y.
/// - `origin` is the centreline point at this `s`.
#[derive(Copy, Clone, Debug)]
pub struct TrackFrame {
    pub s_m: f32,
    pub origin: glam::Vec3,
    pub right: glam::Vec3,
    #[allow(unused)]
    pub up: glam::Vec3,
    pub forward: glam::Vec3,
}

/// A complete closed-loop track.
///
/// For now this is deliberately simple:
/// a rounded rectangle made from straight segments and circular arcs.
///
/// Later, tunnel chunks, lights, rails, walkways, and the cab camera
/// will all attach to this one source of truth.
#[derive(Debug)]
pub struct Track {
    segments: Vec<TrackSegment>,
    length_m: f32,
}

/// One piece of the track.
///
/// Important detail:
/// each segment stores its absolute start pose, not a transform relative
/// to the previous frame. That means sampling `s=3000m` does not require
/// applying 3000 tiny transforms and accumulating numerical drift.
#[derive(Copy, Clone, Debug)]
struct TrackSegment {
    start_s_m: f32,
    length_m: f32,
    start_origin: glam::Vec3,
    start_heading_rad: f32,
    kind: TrackSegmentKind,
}

#[derive(Copy, Clone, Debug)]
enum TrackSegmentKind {
    /// Constant heading.
    Straight,

    /// Circular horizontal curve.
    ///
    /// `turn_sign = 1.0` means heading increases.
    /// With our convention, four positive quarter turns create a loop.
    Arc { radius_m: f32, turn_sign: f32 },
}

impl Track {
    pub fn metro_loop() -> Self {
        // Keep this intentionally boring and inspectable.
        // The showcase does not need a general spline editor yet.
        let radius_m = CURVE_RADIUS_METRES;
        let long_straight_m = STRAIGHT_LENGTH_METRES;

        // One quarter-circle arc length:
        //
        // arc_length = radius * angle
        // angle = pi / 2
        let quarter_arc_m = 0.5 * std::f32::consts::PI * radius_m;

        // We want the full loop to be exactly 5000m.
        // The remaining distance becomes two shorter straights.
        let short_straight_m =
            (TRACK_LENGTH_METRES - 2.0 * long_straight_m - 4.0 * quarter_arc_m) * 0.5;

        assert!(short_straight_m > 0.0);

        let mut segments = Vec::new();

        // These variables describe the start of the next segment.
        // After adding a segment, we advance them to that segment's end.
        let mut start_s_m = 0.0;
        let mut start_origin = glam::Vec3::ZERO;
        let mut start_heading_rad = 0.0;

        let mut push_segment = |length_m: f32, kind: TrackSegmentKind| {
            let segment = TrackSegment {
                start_s_m,
                length_m,
                start_origin,
                start_heading_rad,
                kind,
            };

            // Compute the end pose analytically.
            // This is not frame-by-frame movement.
            let (end_origin, end_heading_rad) = segment.sample_pose(length_m);

            segments.push(segment);

            start_s_m += length_m;
            start_origin = end_origin;
            start_heading_rad = end_heading_rad;
        };

        // Shape:
        //
        //   long straight
        //       + quarter arc
        //       + short straight
        //       + quarter arc
        //       + long straight
        //       + quarter arc
        //       + short straight
        //       + quarter arc
        //
        // This forms a closed rounded rectangle.
        push_segment(long_straight_m, TrackSegmentKind::Straight);
        push_segment(
            quarter_arc_m,
            TrackSegmentKind::Arc {
                radius_m,
                turn_sign: 1.0,
            },
        );
        push_segment(short_straight_m, TrackSegmentKind::Straight);
        push_segment(
            quarter_arc_m,
            TrackSegmentKind::Arc {
                radius_m,
                turn_sign: 1.0,
            },
        );
        push_segment(long_straight_m, TrackSegmentKind::Straight);
        push_segment(
            quarter_arc_m,
            TrackSegmentKind::Arc {
                radius_m,
                turn_sign: 1.0,
            },
        );
        push_segment(short_straight_m, TrackSegmentKind::Straight);
        push_segment(
            quarter_arc_m,
            TrackSegmentKind::Arc {
                radius_m,
                turn_sign: 1.0,
            },
        );

        debug_assert!((start_s_m - TRACK_LENGTH_METRES).abs() < 0.01);

        Self {
            segments,
            length_m: TRACK_LENGTH_METRES,
        }
    }

    /// Sample the track at distance `s_m`.
    ///
    /// This is the key API.
    /// Everything else in the tunnel demo should eventually ask this:
    ///
    ///     "At this track distance, where am I and which way is forward?"
    pub fn sample(&self, s_m: f32) -> TrackFrame {
        let wrapped_s_m = s_m.rem_euclid(self.length_m);

        let segment = self
            .segments
            .iter()
            .rev()
            .find(|segment| wrapped_s_m >= segment.start_s_m)
            .expect("track must contain at least one segment");

        let local_s_m = wrapped_s_m - segment.start_s_m;
        segment.sample(local_s_m, wrapped_s_m)
    }

    /// Temporary debug helper for this milestone.
    pub fn log_debug_samples(&self) {
        for s_m in [
            0.0,
            250.0,
            1_200.0,
            1_500.0,
            2_500.0,
            3_750.0,
            self.length_m - 0.1,
        ] {
            let frame = self.sample(s_m);

            log::info!(
                "track: s={:.1} pos=({:.1}, {:.1}, {:.1}) fwd=({:.3}, {:.3}, {:.3}) right_dot_fwd={:.3}",
                frame.s_m,
                frame.origin.x,
                frame.origin.y,
                frame.origin.z,
                frame.forward.x,
                frame.forward.y,
                frame.forward.z,
                frame.right.dot(frame.forward),
            );
        }

        let start = self.sample(0.0);
        let wrapped = self.sample(self.length_m);

        log::info!(
            "track: wrap delta from s=5000m to s=0m is {:.6}m",
            start.origin.distance(wrapped.origin)
        );
    }
}

impl TrackSegment {
    fn sample(&self, local_s_m: f32, global_s_m: f32) -> TrackFrame {
        let (origin, heading_rad) = self.sample_pose(local_s_m);
        frame_from_heading(global_s_m, origin, heading_rad)
    }

    /// Return the position and heading inside this one segment.
    ///
    /// This is analytic:
    /// - straight segments use `origin + forward * distance`
    /// - arcs use circular-arc equations
    fn sample_pose(&self, local_s_m: f32) -> (glam::Vec3, f32) {
        let local_s_m = local_s_m.clamp(0.0, self.length_m);

        match self.kind {
            TrackSegmentKind::Straight => {
                let forward = forward_from_heading(self.start_heading_rad);

                (
                    self.start_origin + forward * local_s_m,
                    self.start_heading_rad,
                )
            }
            TrackSegmentKind::Arc {
                radius_m,
                turn_sign,
            } => {
                let curvature = turn_sign / radius_m;
                let heading_rad = self.start_heading_rad + curvature * local_s_m;

                let (sin0, cos0) = self.start_heading_rad.sin_cos();
                let (sin1, cos1) = heading_rad.sin_cos();

                // Integrated circular motion in x/z.
                // y stays zero because this first track is flat.
                let dx = (cos0 - cos1) / curvature;
                let dz = (sin1 - sin0) / curvature;

                (self.start_origin + glam::vec3(dx, 0.0, dz), heading_rad)
            }
        }
    }
}

fn frame_from_heading(s_m: f32, origin: glam::Vec3, heading_rad: f32) -> TrackFrame {
    let forward = forward_from_heading(heading_rad);
    let up = glam::Vec3::Y;

    // With forward=+Z and up=+Y, this gives right=+X.
    let right = up.cross(forward).normalize();

    TrackFrame {
        s_m,
        origin,
        right,
        up,
        forward,
    }
}

fn forward_from_heading(heading_rad: f32) -> glam::Vec3 {
    let (sin_heading, cos_heading) = heading_rad.sin_cos();

    // heading 0 means +Z.
    // heading pi/2 means +X.
    glam::vec3(sin_heading, 0.0, cos_heading).normalize()
}
