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

pub fn log_tunnel_mesh_debug(track: &Track) {
    let params = TunnelMeshParams::default();
    let chunk_a = generate_tunnel_shell(track, 1_190.0, 20.0, params);
    let chunk_b = generate_tunnel_shell(track, 1_210.0, 20.0, TunnelMeshParams::default());

    let profile_count = TunnelMeshParams::default().profile_segments + 1;
    let last_a = chunk_a.vertices.len() - profile_count;

    let mut max_seam_error_m: f32 = 0.0;
    for i in 0..profile_count {
        let pa = chunk_a.vertices[last_a + i].position;
        let pb = chunk_b.vertices[i].position;
        max_seam_error_m = max_seam_error_m.max(pa.distance(pb));
    }

    log::info!(
        "tunnel mesh: vertices={} indices={} seam_error={:.6}m",
        chunk_a.vertices.len(),
        chunk_a.indices.len(),
        max_seam_error_m,
    );
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
