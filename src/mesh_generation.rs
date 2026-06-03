use crate::track::Track;
use lazy_vulkan_gltf::Vertex;

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

pub const TUNNEL_BAY_LENGTH_METRES: f32 = 10.0;

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
    let ring_spacing_m = 2.0;
    let half_width_m = 1.75;
    let slab_y_m = -1.08;

    let ring_count = (length_m / ring_spacing_m).ceil() as usize + 1;

    let mut vertices = Vec::with_capacity(ring_count * 2);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        for (side_index, x_m) in [-half_width_m, half_width_m].into_iter().enumerate() {
            let position = frame.origin + frame.right * x_m + frame.up * slab_y_m;
            let normal = frame.up;
            let uv = glam::vec2(side_index as f32 * half_width_m * 2.0, s_m);

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

    // Right side of tunnel, in track space.
    let inner_x_m = 2.05;
    let outer_x_m = 3.05;
    let top_y_m = -0.78;
    let bottom_y_m = -1.08;

    let ring_count = (length_m / ring_spacing_m).ceil() as usize + 1;
    let verts_per_ring = 4;

    let mut vertices = Vec::with_capacity(ring_count * verts_per_ring);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        let corners = [
            (inner_x_m, top_y_m, frame.up),
            (outer_x_m, top_y_m, frame.up),
            (outer_x_m, bottom_y_m, frame.right),
            (inner_x_m, bottom_y_m, -frame.right),
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

pub const LED_TUBE_LENGTH_METRES: f32 = 4.0;
pub const LED_TUBE_X_METRES: f32 = 0.0;
pub const LED_TUBE_Y_METRES: f32 = 3.52;
pub const LED_TUBE_HALF_WIDTH_METRES: f32 = 0.09;
pub const LED_TUBE_HALF_HEIGHT_METRES: f32 = 0.035;
pub const LED_TUBE_LIGHT_RADIUS_METRES: f32 = 7.0;
pub const LED_TUBE_LIGHT_INTENSITY: f32 = 6.0;

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
}
