mod demo_state;
mod graphics;
mod track;

use crate::{
    demo_state::DemoState,
    graphics::{RTRenderer, RenderState, RenderStateFamily, compile_shaders},
    track::Track,
};
use lazy_vulkan::LazyVulkan;
use std::time::Instant;
use winit::{application::ApplicationHandler, window::WindowAttributes};

const CORRIDOR_REPEAT_COUNT: usize = 9;
const CORRIDOR_SPACING_METRES: f32 = 11.0;
const CAMERA_SPEED_METRES_PER_SECOND: f32 = 2.5;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Hello RT"))
            .unwrap();

        let mut lazy_vulkan = LazyVulkan::from_window(&window);
        let renderer = RTRenderer::new(&mut lazy_vulkan.renderer);
        lazy_vulkan.add_sub_renderer(Box::new(renderer));

        let track = Track::metro_loop();
        track.log_debug_samples();

        self.state = Some(State {
            window,
            lazy_vulkan,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            demo_state: DemoState::default(),
            track,
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
                let dt = (now - state.last_frame_time).as_secs_f32();
                state.last_frame_time = now;
                state.demo_state.update(dt);
                let _ = state.track.sample(state.demo_state.track_s_m);

                state.lazy_vulkan.draw(&RenderState {
                    elapsed_seconds: state.start_time.elapsed().as_secs_f32(),
                    demo_state: &state.demo_state,
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
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

fn main() {
    env_logger::init();
    compile_shaders();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
