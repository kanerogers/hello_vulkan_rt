mod demo_state;
mod graphics;
pub mod material_loader;
mod mesh_generation;
mod rt_renderer;
mod track;

use crate::{
    demo_state::DemoState,
    graphics::{RenderState, RenderStateFamily, compile_shaders},
    rt_renderer::RTRenderer,
    track::Track,
};
use lazy_vulkan::LazyVulkan;
use std::time::Instant;
use winit::{application::ApplicationHandler, window::WindowAttributes};

const FIXED_TIMESTEP_S: f32 = 1.0 / 120.0;
const MAX_FRAME_TIME_S: f32 = 0.25;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Hello RT")
                    .with_maximized(true),
            )
            .unwrap();

        let mut lazy_vulkan = LazyVulkan::from_window(&window);
        let track = Track::metro_loop();
        let renderer = RTRenderer::new(&mut lazy_vulkan.renderer, &track);
        lazy_vulkan.add_sub_renderer(Box::new(renderer));

        track.log_debug_samples();

        self.state = Some(State {
            window,
            lazy_vulkan,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            demo_state: DemoState::default(),
            track,
            fixed_time_accumulator_s: 0.0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        use winit::event::WindowEvent;
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let frame_time_s = (now - state.last_frame_time)
                    .as_secs_f32()
                    .min(MAX_FRAME_TIME_S);

                state.last_frame_time = now;
                state.fixed_time_accumulator_s += frame_time_s;

                while state.fixed_time_accumulator_s >= FIXED_TIMESTEP_S {
                    state.demo_state.update(FIXED_TIMESTEP_S);
                    state.fixed_time_accumulator_s -= FIXED_TIMESTEP_S;
                }

                let train_current_frame = state.track.sample(state.demo_state.track_s_m);

                state.lazy_vulkan.draw(&RenderState {
                    elapsed_seconds: state.start_time.elapsed().as_secs_f32(),
                    demo_state: &state.demo_state,
                    train_current_frame: train_current_frame,
                });
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let Some(state) = &mut self.state else {
            return;
        };

        state.window.request_redraw();
    }
}

struct State {
    #[allow(unused)]
    window: winit::window::Window,
    lazy_vulkan: LazyVulkan<RenderStateFamily>,
    start_time: Instant,
    last_frame_time: Instant,
    demo_state: DemoState,
    track: Track,
    fixed_time_accumulator_s: f32,
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

fn main() {
    use winit::platform::x11::EventLoopBuilderExtX11;
    env_logger::init();
    compile_shaders();
    let event_loop = winit::event_loop::EventLoop::builder()
        .with_x11()
        .build()
        .unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
