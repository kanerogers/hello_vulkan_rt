use crate::track::Track;
use lazy_vulkan_gltf::Vertex;

pub const TUNNEL_BAY_LENGTH_METRES: f32 = 10.0;

const SLAB_BED_RING_SPACING_M: f32 = 2.0;
const SLAB_BED_HALF_WIDTH_M: f32 = 2.0;
const SLAB_BED_Y_M: f32 = -1.08;

pub const SERVICE_WALKWAY_INNER_X_METRES: f32 = 2.05;
pub const SERVICE_WALKWAY_OUTER_X_METRES: f32 = 3.05;
pub const SERVICE_WALKWAY_TOP_Y_METRES: f32 = -0.78;
pub const SERVICE_WALKWAY_BOTTOM_Y_METRES: f32 = -1.08;

pub const LED_TUBE_LENGTH_METRES: f32 = 4.0;
pub const LED_TUBE_X_METRES: f32 = 0.0;
pub const LED_TUBE_Y_METRES: f32 = 4.02;
pub const LED_TUBE_HALF_WIDTH_METRES: f32 = 0.09;
pub const LED_TUBE_HALF_HEIGHT_METRES: f32 = 0.035;
pub const LED_TUBE_LIGHT_RADIUS_METRES: f32 = 6.7;
pub const LED_TUBE_LIGHT_INTENSITY: f32 = 5.0;

pub const LOWER_STRIP_LENGTH_METRES: f32 = TUNNEL_BAY_LENGTH_METRES;
pub const LOWER_STRIP_HALF_WIDTH_METRES: f32 = 0.015;
pub const LOWER_STRIP_HALF_HEIGHT_METRES: f32 = 0.025;
pub const LOWER_STRIP_X_METRES: f32 =
    SERVICE_WALKWAY_INNER_X_METRES - LOWER_STRIP_HALF_WIDTH_METRES;
pub const LOWER_STRIP_Y_METRES: f32 =
    (SERVICE_WALKWAY_TOP_Y_METRES + SERVICE_WALKWAY_BOTTOM_Y_METRES) * 0.5;
pub const LOWER_STRIP_TILE_PITCH_METRES: f32 = 0.40;
pub const LOWER_STRIP_TILE_GAP_METRES: f32 = 0.26;
pub const LOWER_STRIP_LIGHT_RADIUS_METRES: f32 = 1.0;
pub const LOWER_STRIP_LIGHT_INTENSITY: f32 = 1.0;

pub struct GeneratedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub struct TunnelMeshParams {
    pub ring_spacing_m: f32,
    pub profile_segments: usize,
    pub radius_m: f32,
    pub centre_y_m: f32,
}

impl Default for TunnelMeshParams {
    fn default() -> Self {
        Self {
            ring_spacing_m: 2.0,
            profile_segments: 32,
            radius_m: 3.8,
            centre_y_m: 0.75,
        }
    }
}

pub fn generate_tunnel_shell(
    track: &Track,
    start_s_m: f32,
    length_m: f32,
    params: TunnelMeshParams,
) -> GeneratedMesh {
    const BOTTOM_PROFILE_SEGMENTS: usize = 8;

    assert!(
        params.profile_segments > BOTTOM_PROFILE_SEGMENTS,
        "tunnel shell needs enough profile segments for arch + bottom"
    );

    let arch_profile_segments = params.profile_segments - BOTTOM_PROFILE_SEGMENTS;

    let ring_count = (length_m / params.ring_spacing_m).ceil() as usize + 1;
    let profile_count = params.profile_segments + 1;

    let arch_start_angle_rad = 210.0_f32.to_radians();
    let arch_end_angle_rad = -30.0_f32.to_radians();
    let arch_angle_span_rad = (arch_end_angle_rad - arch_start_angle_rad).abs();
    let arch_length_metres = params.radius_m * arch_angle_span_rad;

    let bottom_left_x_metres = params.radius_m * arch_start_angle_rad.cos();
    let bottom_right_x_metres = params.radius_m * arch_end_angle_rad.cos();
    let bottom_y_metres = params.centre_y_m + params.radius_m * arch_start_angle_rad.sin();

    let mut vertices = Vec::with_capacity(ring_count * profile_count);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        for profile_index in 0..profile_count {
            let (position, normal, uv) = if profile_index <= arch_profile_segments {
                let arch_t = profile_index as f32 / arch_profile_segments as f32;
                let angle_rad = lerp(arch_start_angle_rad, arch_end_angle_rad, arch_t);

                let local_x_metres = params.radius_m * angle_rad.cos();
                let local_y_metres = params.centre_y_m + params.radius_m * angle_rad.sin();

                let position =
                    frame.origin + frame.right * local_x_metres + frame.up * local_y_metres;

                let profile_outward =
                    glam::vec2(local_x_metres, local_y_metres - params.centre_y_m).normalize();

                let normal =
                    (-frame.right * profile_outward.x - frame.up * profile_outward.y).normalize();

                let uv = glam::vec2(arch_length_metres * arch_t, s_m);

                (position, normal, uv)
            } else {
                let bottom_index = profile_index - arch_profile_segments;
                let bottom_t = bottom_index as f32 / BOTTOM_PROFILE_SEGMENTS as f32;

                let local_x_metres = lerp(bottom_right_x_metres, bottom_left_x_metres, bottom_t);
                let local_y_metres = bottom_y_metres;

                let position =
                    frame.origin + frame.right * local_x_metres + frame.up * local_y_metres;

                let bottom_distance_metres = (bottom_right_x_metres - local_x_metres).abs();
                let uv = glam::vec2(arch_length_metres + bottom_distance_metres, s_m);

                (position, frame.up, uv)
            };

            vertices.push(Vertex::new(position, normal, Some(uv)));
        }
    }

    for ring_index in 0..(ring_count - 1) {
        for profile_index in 0..params.profile_segments {
            let a = (ring_index * profile_count + profile_index) as u32;
            let b = a + 1;
            let c = a + profile_count as u32;
            let d = c + 1;

            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    GeneratedMesh { vertices, indices }
}

pub fn generate_slab_bed(track: &Track, start_s_m: f32, length_m: f32) -> GeneratedMesh {
    let ring_count = (length_m / SLAB_BED_RING_SPACING_M).ceil() as usize + 1;

    let mut vertices = Vec::with_capacity(ring_count * 2);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        for (side_index, x_m) in [-SLAB_BED_HALF_WIDTH_M, SLAB_BED_HALF_WIDTH_M]
            .into_iter()
            .enumerate()
        {
            let position = frame.origin + frame.right * x_m + frame.up * SLAB_BED_Y_M;
            let normal = frame.up;
            let uv = glam::vec2(side_index as f32 * SLAB_BED_HALF_WIDTH_M * 2.0, s_m);

            vertices.push(Vertex::new(position, normal, Some(uv)));
        }
    }

    for ring_index in 0..(ring_count - 1) {
        let a = (ring_index * 2) as u32;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;

        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }

    GeneratedMesh { vertices, indices }
}

pub fn generate_rails(track: &Track, start_s_m: f32, length_m: f32) -> GeneratedMesh {
    let ring_spacing_m = 1.0;
    let rail_gauge_m = 1.435;
    let rail_half_width_m = 0.055;
    let rail_height_m = 0.13;
    let rail_base_y_m = -1.02;

    let rail_x_offsets = [-rail_gauge_m * 0.5, rail_gauge_m * 0.5];
    let ring_count = (length_m / ring_spacing_m).ceil() as usize + 1;
    let verts_per_ring = 8;

    let mut vertices = Vec::with_capacity(ring_count * verts_per_ring);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        for rail_x_m in rail_x_offsets {
            let corners = [
                (-rail_half_width_m, 0.0, -frame.up),
                (rail_half_width_m, 0.0, -frame.up),
                (rail_half_width_m, rail_height_m, frame.up),
                (-rail_half_width_m, rail_height_m, frame.up),
            ];

            for (local_dx_m, local_y_m, normal) in corners {
                let position = frame.origin
                    + frame.right * (rail_x_m + local_dx_m)
                    + frame.up * (rail_base_y_m + local_y_m);

                let uv = glam::vec2(local_dx_m + rail_half_width_m, s_m);
                vertices.push(Vertex::new(position, normal, Some(uv)));
            }
        }
    }

    for ring_index in 0..(ring_count - 1) {
        for rail_index in 0..2 {
            let base0 = (ring_index * verts_per_ring + rail_index * 4) as u32;
            let base1 = base0 + verts_per_ring as u32;

            // left side, top, right side. Bottom is omitted; it sits on the slab.
            for (a, b) in [(0, 3), (3, 2), (2, 1)] {
                let v0 = base0 + a;
                let v1 = base0 + b;
                let v2 = base1 + a;
                let v3 = base1 + b;

                indices.extend_from_slice(&[v0, v2, v1, v1, v2, v3]);
            }
        }
    }

    GeneratedMesh { vertices, indices }
}

pub fn generate_service_walkway(track: &Track, start_s_m: f32, length_m: f32) -> GeneratedMesh {
    let ring_spacing_m = 2.0;

    let ring_count = (length_m / ring_spacing_m).ceil() as usize + 1;
    let verts_per_ring = 4;

    let mut vertices = Vec::with_capacity(ring_count * verts_per_ring);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        let corners = [
            (
                SERVICE_WALKWAY_INNER_X_METRES,
                SERVICE_WALKWAY_TOP_Y_METRES,
                frame.up,
            ),
            (
                SERVICE_WALKWAY_OUTER_X_METRES,
                SERVICE_WALKWAY_TOP_Y_METRES,
                frame.up,
            ),
            (
                SERVICE_WALKWAY_OUTER_X_METRES,
                SERVICE_WALKWAY_BOTTOM_Y_METRES,
                frame.right,
            ),
            (
                SERVICE_WALKWAY_INNER_X_METRES,
                SERVICE_WALKWAY_BOTTOM_Y_METRES,
                -frame.right,
            ),
        ];

        for (corner_index, (x_m, y_m, normal)) in corners.into_iter().enumerate() {
            let position = frame.origin + frame.right * x_m + frame.up * y_m;
            let uv = glam::vec2(corner_index as f32, s_m);

            vertices.push(Vertex::new(position, normal, Some(uv)));
        }
    }

    for ring_index in 0..(ring_count - 1) {
        let base0 = (ring_index * verts_per_ring) as u32;
        let base1 = base0 + verts_per_ring as u32;

        // top, outer face, inner face. Bottom omitted.
        for (a, b) in [(0, 1), (1, 2), (3, 0)] {
            let v0 = base0 + a;
            let v1 = base0 + b;
            let v2 = base1 + a;
            let v3 = base1 + b;

            indices.extend_from_slice(&[v0, v2, v1, v1, v2, v3]);
        }
    }

    GeneratedMesh { vertices, indices }
}

pub fn generate_cable_tray(track: &Track, start_s_m: f32, length_m: f32) -> GeneratedMesh {
    let ring_spacing_m = 2.0;

    let center_x_m = 3.15;
    let center_y_m = 0.55;
    let half_width_m = 0.22;
    let half_height_m = 0.08;

    let ring_count = (length_m / ring_spacing_m).ceil() as usize + 1;
    let verts_per_ring = 4;

    let mut vertices = Vec::with_capacity(ring_count * verts_per_ring);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        let corners = [
            (-half_width_m, -half_height_m, -frame.right),
            (half_width_m, -half_height_m, -frame.up),
            (half_width_m, half_height_m, frame.right),
            (-half_width_m, half_height_m, frame.up),
        ];

        for (corner_index, (dx_m, dy_m, normal)) in corners.into_iter().enumerate() {
            let position =
                frame.origin + frame.right * (center_x_m + dx_m) + frame.up * (center_y_m + dy_m);

            let uv = glam::vec2(corner_index as f32, s_m);
            vertices.push(Vertex::new(position, normal, Some(uv)));
        }
    }

    for ring_index in 0..(ring_count - 1) {
        let base0 = (ring_index * verts_per_ring) as u32;
        let base1 = base0 + verts_per_ring as u32;

        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            let v0 = base0 + a;
            let v1 = base0 + b;
            let v2 = base1 + a;
            let v3 = base1 + b;

            indices.extend_from_slice(&[v0, v2, v1, v1, v2, v3]);
        }
    }

    GeneratedMesh { vertices, indices }
}

#[derive(Copy, Clone, Debug)]
pub struct LedTubeFixture {
    pub start_s_metres: f32,
    pub length_metres: f32,
    pub x_metres: f32,
    pub y_metres: f32,
    pub half_width_metres: f32,
    pub half_height_metres: f32,
    #[allow(unused)]
    pub colour: glam::Vec3,
    #[allow(unused)]
    pub intensity: f32,
    #[allow(unused)]
    pub radius_metres: f32,
}

pub fn generate_led_tube_fixtures(track_length_metres: f32) -> Vec<LedTubeFixture> {
    let fixture_count = (track_length_metres / TUNNEL_BAY_LENGTH_METRES).floor() as usize;

    (0..fixture_count)
        .map(|fixture_index| {
            let start_s_metres = fixture_index as f32 * TUNNEL_BAY_LENGTH_METRES;
            let intensity = led_tube_intensity_for_s(start_s_metres);
            LedTubeFixture {
                start_s_metres,
                length_metres: LED_TUBE_LENGTH_METRES,
                x_metres: LED_TUBE_X_METRES,
                y_metres: LED_TUBE_Y_METRES,
                half_width_metres: LED_TUBE_HALF_WIDTH_METRES,
                half_height_metres: LED_TUBE_HALF_HEIGHT_METRES,
                colour: glam::vec3(0.65, 0.82, 1.0),
                intensity,
                radius_metres: LED_TUBE_LIGHT_RADIUS_METRES,
            }
        })
        .collect()
}

fn led_tube_intensity_for_s(start_s_metres: f32) -> f32 {
    let zone_index = (start_s_metres / 250.0).floor() as u32;

    match zone_index % 5 {
        0 => LED_TUBE_LIGHT_INTENSITY,
        1 => LED_TUBE_LIGHT_INTENSITY * 0.45,
        2 => LED_TUBE_LIGHT_INTENSITY * 0.85,
        3 => LED_TUBE_LIGHT_INTENSITY * 0.25,
        _ => LED_TUBE_LIGHT_INTENSITY * 1.15,
    }
}

pub fn generate_led_tubes(track: &Track, fixtures: &[LedTubeFixture]) -> GeneratedMesh {
    let mut vertices = Vec::with_capacity(fixtures.len() * 8);
    let mut indices = Vec::with_capacity(fixtures.len() * 24);

    for fixture in fixtures {
        let fixture_start_s_metres = fixture.start_s_metres;
        let fixture_end_s_metres = fixture.start_s_metres + fixture.length_metres;

        let frame0 = track.sample(fixture_start_s_metres);
        let frame1 = track.sample(fixture_end_s_metres);

        let base = vertices.len() as u32;

        let corners = [
            (-fixture.half_width_metres, -fixture.half_height_metres),
            (fixture.half_width_metres, -fixture.half_height_metres),
            (fixture.half_width_metres, fixture.half_height_metres),
            (-fixture.half_width_metres, fixture.half_height_metres),
        ];

        for frame in [frame0, frame1] {
            for (corner_index, (dx_metres, dy_metres)) in corners.into_iter().enumerate() {
                let position = frame.origin
                    + frame.right * (fixture.x_metres + dx_metres)
                    + frame.up * (fixture.y_metres + dy_metres);

                let normal = match corner_index {
                    0 => (-frame.right - frame.up).normalize(),
                    1 => (frame.right - frame.up).normalize(),
                    2 => (frame.right + frame.up).normalize(),
                    _ => (-frame.right + frame.up).normalize(),
                };

                let uv = glam::vec2(corner_index as f32, fixture_start_s_metres);
                vertices.push(Vertex::new(position, normal, Some(uv)));
            }
        }

        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            let v0 = base + a;
            let v1 = base + b;
            let v2 = base + 4 + a;
            let v3 = base + 4 + b;

            indices.extend_from_slice(&[v0, v2, v1, v1, v2, v3]);
        }
    }

    GeneratedMesh { vertices, indices }
}

#[allow(unused)]
pub fn generate_shadow_debug_blockers(track: &Track) -> GeneratedMesh {
    const BLOCKER_HALF_WIDTH_METRES: f32 = 0.75;
    const BLOCKER_HALF_HEIGHT_METRES: f32 = 0.50;
    const BLOCKER_HALF_DEPTH_METRES: f32 = 0.75;

    // Centre of cuboid, in track space.
    // y=0.5m puts it floating just above the track/slab area.
    const BLOCKER_X_METRES: f32 = 0.0;
    const BLOCKER_Y_METRES: f32 = 0.50;

    let blocker_s_metres = [25.0, 55.0, 80.0, 125.0, 170.0];

    let mut vertices = Vec::with_capacity(blocker_s_metres.len() * 24);
    let mut indices = Vec::with_capacity(blocker_s_metres.len() * 36);

    for s_metres in blocker_s_metres {
        let frame = track.sample(s_metres);
        let base = vertices.len() as u32;

        let mut push_vertex =
            |x_metres: f32, y_metres: f32, z_metres: f32, normal: glam::Vec3, uv: glam::Vec2| {
                let position = frame.origin
                    + frame.right * (BLOCKER_X_METRES + x_metres)
                    + frame.up * (BLOCKER_Y_METRES + y_metres)
                    + frame.forward * z_metres;

                vertices.push(Vertex::new(position, normal, Some(uv)));
            };

        let faces = [
            // normal, four corners
            (
                -frame.forward,
                [
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
            (
                frame.forward,
                [
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
            (
                -frame.right,
                [
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
            (
                frame.right,
                [
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
            (
                frame.up,
                [
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
            (
                -frame.up,
                [
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        -BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                    (
                        -BLOCKER_HALF_WIDTH_METRES,
                        -BLOCKER_HALF_HEIGHT_METRES,
                        BLOCKER_HALF_DEPTH_METRES,
                    ),
                ],
            ),
        ];

        for (face_index, (normal, corners)) in faces.into_iter().enumerate() {
            let face_base = base + face_index as u32 * 4;

            for (corner_index, (x_metres, y_metres, z_metres)) in corners.into_iter().enumerate() {
                let uv = glam::vec2(corner_index as f32, face_index as f32);
                push_vertex(x_metres, y_metres, z_metres, normal, uv);
            }

            indices.extend_from_slice(&[
                face_base,
                face_base + 1,
                face_base + 2,
                face_base,
                face_base + 2,
                face_base + 3,
            ]);
        }
    }

    GeneratedMesh { vertices, indices }
}

#[derive(Copy, Clone, Debug)]
pub struct LowerStripFixture {
    pub start_s_metres: f32,
    pub length_metres: f32,
    pub x_metres: f32,
    pub y_metres: f32,
    pub half_width_metres: f32,
    pub half_height_metres: f32,
    #[allow(unused)]
    pub colour: glam::Vec3,
    #[allow(unused)]
    pub intensity: f32,
    #[allow(unused)]
    pub radius_metres: f32,
}

pub fn generate_lower_strip_fixtures(track_length_metres: f32) -> Vec<LowerStripFixture> {
    let fixture_count = (track_length_metres / TUNNEL_BAY_LENGTH_METRES).floor() as usize;

    (0..fixture_count)
        .map(|fixture_index| {
            let start_s_metres = fixture_index as f32 * TUNNEL_BAY_LENGTH_METRES;

            LowerStripFixture {
                start_s_metres,
                length_metres: LOWER_STRIP_LENGTH_METRES,
                x_metres: LOWER_STRIP_X_METRES,
                y_metres: LOWER_STRIP_Y_METRES,
                half_width_metres: LOWER_STRIP_HALF_WIDTH_METRES,
                half_height_metres: LOWER_STRIP_HALF_HEIGHT_METRES,
                colour: glam::vec3(0.25, 0.55, 1.0),
                intensity: LOWER_STRIP_LIGHT_INTENSITY,
                radius_metres: LOWER_STRIP_LIGHT_RADIUS_METRES,
            }
        })
        .collect()
}

pub fn generate_lower_strip_lights(track: &Track, fixtures: &[LowerStripFixture]) -> GeneratedMesh {
    let tile_count: usize = fixtures
        .iter()
        .map(|fixture| lower_strip_tile_count(fixture.length_metres))
        .sum();

    let mut vertices = Vec::with_capacity(tile_count * 24);
    let mut indices = Vec::with_capacity(tile_count * 36);

    for fixture in fixtures {
        for tile in lower_strip_tiles(fixture) {
            push_lower_strip_tile(
                track,
                fixture,
                tile.start_s_metres,
                tile.end_s_metres,
                &mut vertices,
                &mut indices,
            );
        }
    }

    GeneratedMesh { vertices, indices }
}

fn lower_strip_tile_count(length_metres: f32) -> usize {
    (length_metres / LOWER_STRIP_TILE_PITCH_METRES)
        .round()
        .max(1.0) as usize
}

fn push_lower_strip_tile(
    track: &Track,
    fixture: &LowerStripFixture,
    tile_start_s_metres: f32,
    tile_end_s_metres: f32,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let frame0 = track.sample(tile_start_s_metres);
    let frame1 = track.sample(tile_end_s_metres);
    let frame_mid = track.sample((tile_start_s_metres + tile_end_s_metres) * 0.5);

    let x0_metres = fixture.x_metres - fixture.half_width_metres;
    let x1_metres = fixture.x_metres + fixture.half_width_metres;
    let y0_metres = fixture.y_metres - fixture.half_height_metres;
    let y1_metres = fixture.y_metres + fixture.half_height_metres;

    let mut push_quad = |corners: [(f32, f32, bool); 4], normal: glam::Vec3| {
        let base = vertices.len() as u32;

        for (x_metres, y_metres, use_end_frame) in corners {
            let frame = if use_end_frame { frame1 } else { frame0 };
            let s_metres = if use_end_frame {
                tile_end_s_metres
            } else {
                tile_start_s_metres
            };

            let position = frame.origin + frame.right * x_metres + frame.up * y_metres;
            let uv = glam::vec2(x_metres, s_metres);
            vertices.push(Vertex::new(position, normal, Some(uv)));
        }

        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    push_quad(
        [
            (x1_metres, y0_metres, false),
            (x1_metres, y0_metres, true),
            (x1_metres, y1_metres, true),
            (x1_metres, y1_metres, false),
        ],
        frame_mid.right,
    );
    push_quad(
        [
            (x0_metres, y0_metres, true),
            (x0_metres, y0_metres, false),
            (x0_metres, y1_metres, false),
            (x0_metres, y1_metres, true),
        ],
        -frame_mid.right,
    );
    push_quad(
        [
            (x0_metres, y1_metres, false),
            (x1_metres, y1_metres, false),
            (x1_metres, y1_metres, true),
            (x0_metres, y1_metres, true),
        ],
        frame_mid.up,
    );
    push_quad(
        [
            (x0_metres, y0_metres, true),
            (x1_metres, y0_metres, true),
            (x1_metres, y0_metres, false),
            (x0_metres, y0_metres, false),
        ],
        -frame_mid.up,
    );
    push_quad(
        [
            (x0_metres, y0_metres, false),
            (x1_metres, y0_metres, false),
            (x1_metres, y1_metres, false),
            (x0_metres, y1_metres, false),
        ],
        -frame_mid.forward,
    );
    push_quad(
        [
            (x1_metres, y0_metres, true),
            (x0_metres, y0_metres, true),
            (x0_metres, y1_metres, true),
            (x1_metres, y1_metres, true),
        ],
        frame_mid.forward,
    );
}

#[derive(Copy, Clone, Debug)]
pub struct LowerStripTile {
    pub start_s_metres: f32,
    pub end_s_metres: f32,
}

pub fn lower_strip_tiles(fixture: &LowerStripFixture) -> Vec<LowerStripTile> {
    let tile_count = lower_strip_tile_count(fixture.length_metres);
    let tile_gap_metres = if tile_count > 1 {
        LOWER_STRIP_TILE_GAP_METRES
    } else {
        0.0
    };

    let tile_length_metres =
        (fixture.length_metres - tile_gap_metres * (tile_count - 1) as f32) / tile_count as f32;

    (0..tile_count)
        .map(|tile_index| {
            let start_s_metres =
                fixture.start_s_metres + tile_index as f32 * (tile_length_metres + tile_gap_metres);

            LowerStripTile {
                start_s_metres,
                end_s_metres: start_s_metres + tile_length_metres,
            }
        })
        .collect()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_shared_ring_error(
        a: &GeneratedMesh,
        b: &GeneratedMesh,
        ring_vertex_count: usize,
    ) -> f32 {
        let last_a = a.vertices.len() - ring_vertex_count;

        let mut max_error_m: f32 = 0.0;
        for i in 0..ring_vertex_count {
            let pa = a.vertices[last_a + i].position;
            let pb = b.vertices[i].position;
            max_error_m = max_error_m.max(pa.distance(pb));
        }

        max_error_m
    }

    #[test]
    fn tunnel_shell_chunks_share_exact_seam_vertices() {
        let track = Track::metro_loop();
        let params = TunnelMeshParams::default();

        let a = generate_tunnel_shell(&track, 1_190.0, 20.0, params);
        let b = generate_tunnel_shell(&track, 1_210.0, 20.0, TunnelMeshParams::default());

        let ring_vertex_count = TunnelMeshParams::default().profile_segments + 1;
        assert!(max_shared_ring_error(&a, &b, ring_vertex_count) < 0.0001);
    }

    #[test]
    fn slab_bed_chunks_share_exact_seam_vertices() {
        let track = Track::metro_loop();

        let a = generate_slab_bed(&track, 1_190.0, 20.0);
        let b = generate_slab_bed(&track, 1_210.0, 20.0);

        assert!(max_shared_ring_error(&a, &b, 2) < 0.0001);
    }

    #[test]
    fn slab_bed_has_expected_topology() {
        let track = Track::metro_loop();
        let mesh = generate_slab_bed(&track, 0.0, 20.0);

        assert_eq!(mesh.vertices.len(), 22);
        assert_eq!(mesh.indices.len(), 60);
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn rail_chunks_share_exact_seam_vertices() {
        let track = Track::metro_loop();

        let a = generate_rails(&track, 1_190.0, 20.0);
        let b = generate_rails(&track, 1_210.0, 20.0);

        assert!(max_shared_ring_error(&a, &b, 8) < 0.0001);
    }

    #[test]
    fn rails_have_expected_topology() {
        let track = Track::metro_loop();
        let mesh = generate_rails(&track, 0.0, 20.0);

        assert_eq!(mesh.vertices.len(), 168);
        assert_eq!(mesh.indices.len(), 720);
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn service_walkway_chunks_share_exact_seam_vertices() {
        let track = Track::metro_loop();

        let a = generate_service_walkway(&track, 1_190.0, 20.0);
        let b = generate_service_walkway(&track, 1_210.0, 20.0);

        assert!(max_shared_ring_error(&a, &b, 4) < 0.0001);
    }

    #[test]
    fn service_walkway_has_expected_topology() {
        let track = Track::metro_loop();
        let mesh = generate_service_walkway(&track, 0.0, 20.0);

        assert_eq!(mesh.vertices.len(), 44);
        assert_eq!(mesh.indices.len(), 180);
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn cable_tray_chunks_share_exact_seam_vertices() {
        let track = Track::metro_loop();

        let a = generate_cable_tray(&track, 1_190.0, 20.0);
        let b = generate_cable_tray(&track, 1_210.0, 20.0);

        assert!(max_shared_ring_error(&a, &b, 4) < 0.0001);
    }

    #[test]
    fn cable_tray_has_expected_topology() {
        let track = Track::metro_loop();
        let mesh = generate_cable_tray(&track, 0.0, 20.0);

        assert_eq!(mesh.vertices.len(), 44);
        assert_eq!(mesh.indices.len(), 240);
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn lower_strip_lights_have_expected_topology() {
        let track = Track::metro_loop();
        let fixtures = generate_lower_strip_fixtures(40.0);
        let mesh = generate_lower_strip_lights(&track, &fixtures);

        let tiles_per_fixture = lower_strip_tile_count(TUNNEL_BAY_LENGTH_METRES);
        let total_tiles = fixtures.len() * tiles_per_fixture;

        assert_eq!(fixtures.len(), 4);
        assert_eq!(fixtures[0].length_metres, TUNNEL_BAY_LENGTH_METRES);
        assert_eq!(
            fixtures[0].x_metres + fixtures[0].half_width_metres,
            SERVICE_WALKWAY_INNER_X_METRES
        );
        assert_eq!(
            fixtures[0].y_metres,
            (SERVICE_WALKWAY_TOP_Y_METRES + SERVICE_WALKWAY_BOTTOM_Y_METRES) * 0.5
        );
        assert_eq!(mesh.vertices.len(), total_tiles * 24);
        assert_eq!(mesh.indices.len(), total_tiles * 36);
        assert_eq!(mesh.indices.len() % 3, 0);
    }
}
