use crate::{mesh_generation, track::Track};

pub const LOWER_STRIP_LIGHTS_PER_BAY: usize = 25;
pub const TUNNEL_LIGHTS_PER_BAY: usize = 1 + LOWER_STRIP_LIGHTS_PER_BAY;

// Lights
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TunnelLight {
    pub position: glam::Vec3,
    pub radius_metres: f32,
    pub colour: glam::Vec3,
    pub intensity: f32,
    pub emission_direction: glam::Vec3,
    pub directional_falloff_power: f32,
}

unsafe impl bytemuck::Zeroable for TunnelLight {}
unsafe impl bytemuck::Pod for TunnelLight {}

pub fn generate_tunnel_lights(
    track: &Track,
    bay_count: usize,
    led_fixtures: &[mesh_generation::LedTubeFixture],
    lower_strip_fixtures: &[mesh_generation::LowerStripFixture],
) -> Vec<TunnelLight> {
    let mut lights = Vec::with_capacity(bay_count * TUNNEL_LIGHTS_PER_BAY);

    debug_assert_eq!(led_fixtures.len(), bay_count);
    debug_assert_eq!(lower_strip_fixtures.len(), bay_count);

    for bay_index in 0..bay_count {
        let led = led_fixtures[bay_index];
        let led_center_s_metres = led.start_s_metres + led.length_metres * 0.5;
        let led_frame = track.sample(led_center_s_metres);

        lights.push(TunnelLight {
            position: led_frame.origin
                + led_frame.right * led.x_metres
                + led_frame.up * led.y_metres,
            radius_metres: led.radius_metres,
            colour: led.colour,
            intensity: led.intensity,
            emission_direction: -led_frame.up,
            directional_falloff_power: 1.5,
        });

        let strip = lower_strip_fixtures[bay_index];
        let strip_tiles = mesh_generation::lower_strip_tiles(&strip);
        debug_assert_eq!(strip_tiles.len(), LOWER_STRIP_LIGHTS_PER_BAY);

        for tile in strip_tiles {
            let center_s_metres = (tile.start_s_metres + tile.end_s_metres) * 0.5;
            let frame = track.sample(center_s_metres);
            let inward_face_x_metres = strip.x_metres - strip.half_width_metres;

            lights.push(TunnelLight {
                position: frame.origin
                    + frame.right * inward_face_x_metres
                    + frame.up * strip.y_metres,
                radius_metres: strip.radius_metres,
                colour: strip.colour,
                intensity: strip.intensity,
                emission_direction: -frame.right,
                directional_falloff_power: 1.5,
            });
        }
    }

    debug_assert_eq!(lights.len(), bay_count * TUNNEL_LIGHTS_PER_BAY);
    lights
}
