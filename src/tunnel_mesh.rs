use crate::track::Track;
use lazy_vulkan_gltf::Vertex;

pub struct TunnelMesh {
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
) -> TunnelMesh {
    let ring_count = (length_m / params.ring_spacing_m).ceil() as usize + 1;
    let profile_count = params.profile_segments + 1;

    let mut vertices = Vec::with_capacity(ring_count * profile_count);
    let mut indices = Vec::new();

    for ring_index in 0..ring_count {
        let ring_t = ring_index as f32 / (ring_count - 1) as f32;
        let s_m = start_s_m + ring_t * length_m;
        let frame = track.sample(s_m);

        for profile_index in 0..profile_count {
            let profile_t = profile_index as f32 / params.profile_segments as f32;

            // Open lower-left to lower-right arch, looking forward.
            // This leaves floor/track-bed geometry for the next milestone.
            let angle_rad = lerp(210.0_f32.to_radians(), -30.0_f32.to_radians(), profile_t);

            let local_x_m = params.radius_m * angle_rad.cos();
            let local_y_m = params.centre_y_m + params.radius_m * angle_rad.sin();

            let position = frame.origin + frame.right * local_x_m + frame.up * local_y_m;

            // Inward-facing analytic normal.
            // At the crown this points down into the tunnel, not outward into rock.
            let profile_outward = glam::vec2(local_x_m, local_y_m - params.centre_y_m).normalize();

            let normal =
                (-frame.right * profile_outward.x - frame.up * profile_outward.y).normalize();

            // UVs are in metres for now:
            // u = distance around arch, v = distance along track.
            let arch_length_m = params.radius_m * (240.0_f32.to_radians()) * profile_t;
            let uv = glam::vec2(arch_length_m, s_m);

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

    TunnelMesh { vertices, indices }
}

pub fn generate_slab_bed(track: &Track, start_s_m: f32, length_m: f32) -> TunnelMesh {
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

    TunnelMesh { vertices, indices }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_shared_ring_error(a: &TunnelMesh, b: &TunnelMesh, ring_vertex_count: usize) -> f32 {
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
}
