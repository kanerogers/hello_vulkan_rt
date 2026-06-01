use crate::{demo_state::DemoState, track::TrackFrame};
use lazy_vulkan::StateFamily;
use std::default::Default;

pub struct RenderState<'a> {
    #[allow(unused)]
    pub elapsed_seconds: f32,
    pub demo_state: &'a DemoState,
    pub train_current_frame: TrackFrame,
}

pub struct RenderStateFamily;

impl StateFamily for RenderStateFamily {
    type For<'a> = RenderState<'a>;
}

pub fn compile_shaders() {
    use shader_slang as slang;

    const SHADERS: &[(&str, &str)] = &[
        ("shaders/raygen.slang", "shaders/raygen.rgen.spv"),
        ("shaders/miss.slang", "shaders/miss.rmiss.spv"),
        ("shaders/closesthit.slang", "shaders/closesthit.rchit.spv"),
        ("shaders/fullscreen.slang", "shaders/fullscreen.vert.spv"),
        ("shaders/tonemapping.slang", "shaders/tonemapping.frag.spv"),
    ];

    let global_session = slang::GlobalSession::new().unwrap();
    let search_path = std::ffi::CString::new("shaders/").unwrap();

    let session_options = slang::CompilerOptions::default()
        .language(slang::SourceLanguage::Slang)
        .optimization(slang::OptimizationLevel::None)
        .debug_information(slang::DebugInfoLevel::Maximal)
        .glsl_force_scalar_layout(true)
        .matrix_layout_column(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("glsl_460"));

    let targets = [target_desc];
    let search_paths = [search_path.as_ptr()];
    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();

    for (input_path, output_path) in SHADERS {
        log::debug!("[SHADERS] Compiling {input_path} to {output_path}");

        let module = session.load_module(input_path).unwrap();
        let entry_point = module.find_entry_point_by_name("main").unwrap();
        let program = session
            .create_composite_component_type(&[module.clone().into(), entry_point.into()])
            .unwrap();
        let linked_program = program.link().unwrap();
        let shader_bytecode = linked_program.entry_point_code(0, 0).unwrap();

        std::fs::write(output_path, shader_bytecode.as_slice()).unwrap();
    }
}
