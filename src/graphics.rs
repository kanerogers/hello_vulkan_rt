use crate::{demo_state::DemoState, track::TrackFrame};
use anyhow::{Context, Result};
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

#[derive(Clone, Copy)]
struct ShaderCompileJob {
    input_path: &'static str,
    output_path: &'static str,
}

const RT_SHADER_COMPILE_JOBS: &[ShaderCompileJob] = &[
    ShaderCompileJob {
        input_path: "shaders/raygen.slang",
        output_path: "shaders/raygen.rgen.spv",
    },
    ShaderCompileJob {
        input_path: "shaders/miss.slang",
        output_path: "shaders/miss.rmiss.spv",
    },
    ShaderCompileJob {
        input_path: "shaders/shadowmiss.slang",
        output_path: "shaders/shadowmiss.rmiss.spv",
    },
    ShaderCompileJob {
        input_path: "shaders/closesthit.slang",
        output_path: "shaders/closesthit.rchit.spv",
    },
];

const GRAPHICS_SHADER_COMPILE_JOBS: &[ShaderCompileJob] = &[
    ShaderCompileJob {
        input_path: "shaders/fullscreen.slang",
        output_path: "shaders/fullscreen.vert.spv",
    },
    ShaderCompileJob {
        input_path: "shaders/tonemapping.slang",
        output_path: "shaders/tonemapping.frag.spv",
    },
];

pub const RT_SHADER_WATCH_PATHS: &[&str] = &[
    "shaders/raygen.slang",
    "shaders/miss.slang",
    "shaders/shadowmiss.slang",
    "shaders/closesthit.slang",
    "shaders/common.slang",
    "shaders/sampling.slang",
];

pub fn compile_shaders() {
    compile_shader_jobs(RT_SHADER_COMPILE_JOBS)
        .and_then(|_| compile_shader_jobs(GRAPHICS_SHADER_COMPILE_JOBS))
        .unwrap();
}

pub fn compile_rt_shaders() -> Result<()> {
    compile_shader_jobs(RT_SHADER_COMPILE_JOBS)
}

fn compile_shader_jobs(shader_jobs: &[ShaderCompileJob]) -> Result<()> {
    use shader_slang as slang;

    let global_session = slang::GlobalSession::new().context("failed to create Slang session")?;
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

    let session = global_session
        .create_session(&session_desc)
        .context("failed to create Slang compile session")?;

    for shader_job in shader_jobs {
        log::debug!(
            "[SHADERS] Compiling {} to {}",
            shader_job.input_path,
            shader_job.output_path
        );

        let module = session
            .load_module(shader_job.input_path)
            .with_context(|| format!("failed to load {}", shader_job.input_path))?;
        let entry_point = module
            .find_entry_point_by_name("main")
            .with_context(|| format!("failed to find main in {}", shader_job.input_path))?;
        let program = session
            .create_composite_component_type(&[module.clone().into(), entry_point.into()])
            .with_context(|| format!("failed to compose {}", shader_job.input_path))?;
        let linked_program = program
            .link()
            .with_context(|| format!("failed to link {}", shader_job.input_path))?;
        let shader_bytecode = linked_program
            .entry_point_code(0, 0)
            .with_context(|| format!("failed to generate SPIR-V for {}", shader_job.input_path))?;

        std::fs::write(shader_job.output_path, shader_bytecode.as_slice())
            .with_context(|| format!("failed to write {}", shader_job.output_path))?;
    }

    Ok(())
}
