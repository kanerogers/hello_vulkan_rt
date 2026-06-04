use std::sync::Arc;

use anyhow::Context;
use lazy_vulkan::{
    BufferAllocation, FULL_IMAGE, LayerInfo, PipelineOptions, SubRenderer,
    vk::{self, Packed24_8},
};
use lazy_vulkan_gltf::{NO_TEXTURE, TextureID};

use crate::{
    demo_state::{DemoState, TRACK_LENGTH_METRES},
    graphics::{RenderState, RenderStateFamily},
    mesh_generation::{
        self, TUNNEL_BAY_LENGTH_METRES, generate_cable_tray, generate_led_tube_fixtures,
        generate_led_tubes, generate_lower_strip_fixtures, generate_lower_strip_lights,
        generate_rails, generate_service_walkway, generate_slab_bed, generate_tunnel_shell,
    },
    track::{Track, TrackFrame},
};

static CLOSEST_HIT_SHADER_PATH: &'static str = "shaders/closesthit.rchit.spv";
static MISS_SHADER_PATH: &'static str = "shaders/miss.rmiss.spv";
static SHADOW_MISS_SHADER_PATH: &'static str = "shaders/shadowmiss.rmiss.spv";
static RAYGEN_SHADER_PATH: &'static str = "shaders/raygen.rgen.spv";
static TONEMAPPING_SHADER_PATH: &'static str = "shaders/tonemapping.frag.spv";
static FULLSCREEN_SHADER_PATH: &'static str = "shaders/fullscreen.vert.spv";

const RAYGEN_INDEX: u32 = 0;
const MISS_INDEX: u32 = 1;
const SHADOW_MISS_INDEX: u32 = 2;
const CLOSEST_HIT_INDEX: u32 = 3;

pub struct RTRenderer {
    context: Arc<lazy_vulkan::Context>,
    image: lazy_vulkan::Image,
    state: Option<RTState>,
    #[allow(unused)]
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    tonemapping_pipeline: lazy_vulkan::Pipeline,
    tonemapping_descriptor_set: vk::DescriptorSet,
    #[allow(unused)]
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    scene_data: SceneData,
}

impl RTRenderer {
    pub fn new(renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>, track: &Track) -> Self {
        // Create our backing image
        let extent = renderer.get_drawable_extent();
        let image = renderer.create_image(
            "RT Target",
            vk::Format::R16G16B16A16_SFLOAT,
            extent,
            &[],
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        );

        // Scene data
        let scene_data = SceneData::new(renderer, track);

        let device = &renderer.context.device;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 10,
            },
        ];

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .unwrap();

        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                            | vk::ShaderStageFlags::MISS_KHR,
                        descriptor_count: 1,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_count: 1,
                        ..Default::default()
                    },
                ]),
                None,
            )
        }
        .unwrap();

        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(std::slice::from_ref(&layout)),
                )
                .unwrap()[0]
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(
                            vk::ShaderStageFlags::RAYGEN_KHR
                                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        )
                        .size(std::mem::size_of::<Registers>() as _)]),
                None,
            )
        }
        .unwrap();

        let context = &renderer.context;

        let pipeline = create_rt_pipeline(pipeline_layout, context);

        let tonemapping_pipeline = renderer.create_pipeline_with_options::<TonemappingRegisters>(
            &r(FULLSCREEN_SHADER_PATH),
            &r(TONEMAPPING_SHADER_PATH),
            PipelineOptions {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            },
        );

        Self {
            context: renderer.context.clone(),
            state: None,
            image,
            descriptor_pool,
            descriptor_set,
            pipeline,
            pipeline_layout,
            tonemapping_pipeline,
            tonemapping_descriptor_set: renderer.descriptors.set,
            scene_data,
        }
    }
}

impl<'a> SubRenderer<'a> for RTRenderer {
    type State = RenderState<'a>;

    fn stage_transfers(
        &mut self,
        _state: &Self::State,
        allocator: &mut lazy_vulkan::Allocator,
        _image_manager: &mut lazy_vulkan::ImageManager,
    ) {
        // No need to rebuild if we already have a state
        if self.state.is_some() {
            return;
        }

        let rt_state = RTState::new(
            &self.context,
            allocator,
            &mut self.scene_data,
            self.pipeline,
        );

        let device = &self.context.device;

        // Update our descriptor sets
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .dst_set(self.descriptor_set)
                        .dst_binding(0)
                        .descriptor_count(1)
                        .push_next(
                            &mut vk::WriteDescriptorSetAccelerationStructureKHR::default()
                                .acceleration_structures(&[rt_state.tlas]),
                        ),
                    vk::WriteDescriptorSet::default()
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .dst_set(self.descriptor_set)
                        .dst_binding(1)
                        .descriptor_count(1)
                        .image_info(&[vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(self.image.view)]),
                ],
                &[],
            )
        };

        self.state = Some(rt_state);
    }

    fn draw_layer(
        &mut self,
        state: &Self::State,
        context: &lazy_vulkan::Context,
        layer_info: LayerInfo,
    ) {
        let demo_state = &state.demo_state;
        let Some(rt_state) = &self.state else { return };

        let device = &context.device;
        let command_buffer = context.draw_command_buffer;
        let drawable = &layer_info.colour_attachment.unwrap();
        let scene_data = &self.scene_data;

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(self.image.handle)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL),
                ]),
            );

            let aspect_ratio = drawable.extent.width as f32 / drawable.extent.height as f32;
            let mut perspective = glam::Mat4::perspective_infinite_reverse_rh(
                60_f32.to_radians(),
                aspect_ratio,
                0.01,
            );

            // wulkankjzk
            perspective.y_axis *= -1.0;

            let view_inverse = camera_view_from_train_frame(state.train_current_frame, demo_state);

            let registers = Registers {
                view_inverse: view_inverse,
                proj_inverse: perspective.inverse(),
                primitive_buffer: scene_data.primitive_buffer.device_address,
                tunnel_bay_length_metres: TUNNEL_BAY_LENGTH_METRES,
                frame: 0, // TODO
                light_buffer: scene_data.light_buffer.device_address,
                light_count: scene_data.light_buffer.len() as u32,
                train_track_s_metres: demo_state.track_s_m,
                track_length_metres: TRACK_LENGTH_METRES,
            };

            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                0,
                bytemuck::bytes_of(&registers),
            );

            context.ray_tracing_pipeline_pfn.cmd_trace_rays(
                command_buffer,
                &rt_state.sbt.gen_region,
                &rt_state.sbt.miss_region,
                &rt_state.sbt.hit_region,
                &rt_state.sbt.call_region,
                drawable.extent.width,
                drawable.extent.height,
                1,
            );

            // rt > fragment shader
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(self.image.handle)
                        .src_access_mask(
                            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
                        )
                        .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                ]),
            );

            context.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(drawable.extent.into())
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(drawable.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        })]),
            );

            self.tonemapping_pipeline
                .update_registers(&TonemappingRegisters {
                    texture_id: self.image.id,
                });

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.tonemapping_pipeline.handle,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.tonemapping_pipeline.layout,
                0,
                &[self.tonemapping_descriptor_set],
                &[],
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);
        };
    }

    fn label(&self) -> &'static str {
        "RT Renderer"
    }
}

/// This is a wrapper around all our scene-specific data.
pub struct SceneData {
    #[allow(unused)]
    vertex_buffer: BufferAllocation<lazy_vulkan_gltf::Vertex>,
    #[allow(unused)]
    index_buffer: BufferAllocation<u32>,
    primitive_buffer: BufferAllocation<Primitive>,
    instance_buffer: BufferAllocation<vk::AccelerationStructureInstanceKHR>,
    scene_primitives: Vec<ScenePrimitive>,
    scene_instances: Vec<SceneInstance>,
    #[allow(unused)]
    tunnel_lights: Vec<TunnelLight>,
    light_buffer: BufferAllocation<TunnelLight>,
}

impl SceneData {
    pub fn new(renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>, track: &Track) -> Self {
        // Create our buffers
        let mut vertex_buffer = renderer.allocator.allocate_buffer(
            20 * 1024 * 1024,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let mut index_buffer = renderer.allocator.allocate_buffer(
            10 * 1024 * 1024,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let mut primitive_buffer = renderer
            .allocator
            .allocate_buffer(10 * 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER);

        let mut light_buffer = renderer
            .allocator
            .allocate_buffer(20 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER);

        // Create the tunnel mesh
        let tunnel_mesh =
            generate_tunnel_shell(track, 0.0, TRACK_LENGTH_METRES, Default::default());

        let tunnel = create_scene_primitive(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            tunnel_mesh,
            glam::vec4(0.55, 0.57, 0.56, 1.0),
        );

        // Generate the slab bed mesh
        let slab_bed_mesh = generate_slab_bed(track, 0.0, TRACK_LENGTH_METRES);
        let slab_bed = create_scene_primitive(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            slab_bed_mesh,
            glam::vec4(0.32, 0.33, 0.33, 1.0),
        );

        // Generate the rails
        let rail_mesh = generate_rails(&track, 0.0, TRACK_LENGTH_METRES);
        let rails = create_scene_primitive(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            rail_mesh,
            glam::vec4(0.10, 0.105, 0.11, 1.0),
        );

        // Generate the service walkway
        let service_walkway_mesh = generate_service_walkway(track, 0.0, TRACK_LENGTH_METRES);
        let service_walkway = create_scene_primitive(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            service_walkway_mesh,
            glam::vec4(0.55, 0.57, 0.56, 1.0),
        );

        // Generate the cable tray
        let cable_tray_mesh = generate_cable_tray(&track, 0.0, TRACK_LENGTH_METRES);
        let cable_tray = create_scene_primitive(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            cable_tray_mesh,
            glam::vec4(0.42, 0.43, 0.41, 1.0),
        );

        // Generate the LED tubes
        let led_tube_fixtures = generate_led_tube_fixtures(TRACK_LENGTH_METRES);
        let led_tube_mesh = generate_led_tubes(track, &led_tube_fixtures);
        let led_tubes = create_scene_primitive_with_emission(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            led_tube_mesh,
            glam::vec4(0.82, 0.90, 1.0, 1.0),
            glam::vec3(0.65, 0.85, 1.0) * 2.0,
        );

        // let blocker_mesh = generate_shadow_debug_blockers(track);
        // let blockers = create_scene_primitive(
        //     renderer,
        //     &mut vertex_buffer,
        //     &mut index_buffer,
        //     blocker_mesh,
        //     glam::vec4(0.05, 0.05, 0.07, 1.0),
        // );

        let lower_strip_fixtures = generate_lower_strip_fixtures(TRACK_LENGTH_METRES);
        let lower_strip_mesh = generate_lower_strip_lights(track, &lower_strip_fixtures);
        let lower_strips = create_scene_primitive_with_emission(
            renderer,
            &mut vertex_buffer,
            &mut index_buffer,
            lower_strip_mesh,
            glam::vec4(0.18, 0.35, 1.0, 1.0),
            glam::vec3(0.15, 0.35, 1.0),
        );

        // Gather our scene primitives
        let scene_primitives = vec![
            tunnel,
            slab_bed,
            rails,
            service_walkway,
            cable_tray,
            led_tubes,
            lower_strips,
        ];

        // Append the data to our primitive buffer
        let primitive_data: Vec<Primitive> = scene_primitives
            .iter()
            .copied()
            .map(
                |ScenePrimitive {
                     indices,
                     vertices,
                     material,
                     index_count: _,
                     vertex_count: _,
                 }| {
                    Primitive {
                        material,
                        index_buffer: indices,
                        vertex_buffer: vertices,
                    }
                },
            )
            .collect();
        primitive_buffer.append(&primitive_data, &mut renderer.allocator);

        // One of each, please.
        let scene_instances = primitive_data
            .iter()
            .enumerate()
            .map(|(i, _)| SceneInstance {
                primitive_index: i,
                world_from_local: glam::Affine3A::default(),
            })
            .collect::<Vec<_>>();

        // Create the instance buffer
        let instance_buffer = renderer
            .allocator
            .allocate_buffer::<vk::AccelerationStructureInstanceKHR>(
                scene_instances.len(),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            );

        // Create our lights
        let tunnel_lights =
            generate_tunnel_lights(track, &led_tube_fixtures, &lower_strip_fixtures);

        // Upload to the light buffer
        light_buffer.append(&tunnel_lights, &mut renderer.allocator);

        Self {
            vertex_buffer,
            index_buffer,
            primitive_buffer,
            instance_buffer,
            scene_primitives,
            scene_instances,
            tunnel_lights,
            light_buffer,
        }
    }
}

fn create_scene_primitive(
    renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>,
    vertex_buffer: &mut BufferAllocation<lazy_vulkan_gltf::Vertex>,
    index_buffer: &mut BufferAllocation<u32>,
    mesh: mesh_generation::GeneratedMesh,
    base_colour_factor: glam::Vec4,
) -> ScenePrimitive {
    create_scene_primitive_with_emission(
        renderer,
        vertex_buffer,
        index_buffer,
        mesh,
        base_colour_factor,
        glam::Vec3::ZERO,
    )
}

fn create_scene_primitive_with_emission(
    renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>,
    vertex_buffer: &mut BufferAllocation<lazy_vulkan_gltf::Vertex>,
    index_buffer: &mut BufferAllocation<u32>,
    mesh: mesh_generation::GeneratedMesh,
    base_colour_factor: glam::Vec4,
    emissive_colour_factor: glam::Vec3,
) -> ScenePrimitive {
    // Upload mesh to buffer
    let indices = index_buffer.tip_address();
    index_buffer.append(&mesh.indices, &mut renderer.allocator);
    let vertices = vertex_buffer.tip_address();
    vertex_buffer.append(&mesh.vertices, &mut renderer.allocator);

    let no_texture: TextureID = NO_TEXTURE.into();

    // Create a simple grey concrete material
    let material = renderer
        .allocator
        .upload_to_slab(&[lazy_vulkan_gltf::GPUMaterial {
            base_colour_factor,
            emissive_colour_factor,

            base_colour_texture: no_texture,
            normal_texture: no_texture,
            metallic_roughness_texture: no_texture,
            ao_texture: no_texture,
        }]);

    ScenePrimitive {
        indices,
        vertices,
        index_count: mesh.indices.len() as u32,
        vertex_count: mesh.vertices.len() as u32,
        material: material.device_address,
    }
}

/// A primitive is, like a glTF primitive, the smallest unit of mesh data.
///
/// It has a material, and some fixed set of vertices, which are indexed by an index buffer.
///
/// This simple abstraction helps us manage our BLASes without needing to dick around too much.
#[derive(Copy, Clone, Debug)]
struct ScenePrimitive {
    // Device pointer to the Index Buffer
    indices: vk::DeviceAddress,

    // Device pointer to the Vertex Buffer
    vertices: vk::DeviceAddress,

    // Counts used for building the BLAS
    index_count: u32,
    vertex_count: u32,

    // Device pointer to the Material
    material: vk::DeviceAddress,
}

/// A scene instance the atomic unit of rendering. It is, basically:
///
/// - A pointer to a [`ScenePrimitive`]
/// - A transform
///
/// That's it.
#[derive(Debug, Clone, Copy)]
struct SceneInstance {
    // Index into `scene_primitives`
    primitive_index: usize,

    // Transform
    world_from_local: glam::Affine3A,
}

/// A bag of useful state
pub struct RTState {
    #[allow(unused)]
    tlas: vk::AccelerationStructureKHR,
    sbt: SBT,
}

impl RTState {
    pub fn new(
        context: &lazy_vulkan::Context,
        allocator: &mut lazy_vulkan::Allocator,
        scene_data: &mut SceneData,
        pipeline: vk::Pipeline,
    ) -> Self {
        let command_buffer = context.draw_command_buffer;

        // Flush any transfers.
        // (TODO): This is kinda yuck.
        //
        // Solutions:
        // 1) `append_now(command_buffer)` <-- does what it says on the tin
        // 2) `on_transferred` <-- add a functor to be called when the transfer has been recorded
        // 3) `Buffer.flush(command_buffer` / `allocator.flush_buffer(buffer, command_buffer)` <-- executes just the transfers for this buffer
        allocator.execute_transfers(command_buffer);

        // First, build up our primitive buffer
        // We start with a map between primitive indices and BLAS indices
        let mut primitive_blas = Vec::with_capacity(scene_data.scene_primitives.len());

        // Then we iterate through the scene primitives and build the BLAS for each one
        for primitive in &scene_data.scene_primitives {
            // Build the BLAS for this primitive
            let blas = build_blas(context, allocator, command_buffer, primitive);

            // Add it to our map
            primitive_blas.push(blas);
        }

        // Next, build our instance buffer
        for instance in &scene_data.scene_instances {
            // Create an instance for this.. instance.
            create_instance(
                context,
                allocator,
                &mut scene_data.instance_buffer,
                &primitive_blas,
                instance,
            );
        }

        // Now, excecute those transfers
        allocator.execute_transfers(command_buffer);

        // Issue a barrier to make sure the data is visible to the acceleration structure
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(scene_data.instance_buffer.handle)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .size(vk::WHOLE_SIZE),
                ]),
            )
        };

        // Now, build the TLAS using the BLASes and instances we generated.
        let tlas = build_tlas(context, allocator, scene_data, command_buffer);

        // Build the SBT
        let sbt = SBT::new(context, allocator, pipeline);

        // Finally, execute the transfer command buffer to upload the SBT data to the GPU
        allocator.execute_transfers(command_buffer);

        RTState { tlas, sbt }
    }
}

fn create_instance(
    context: &lazy_vulkan::Context,
    allocator: &mut lazy_vulkan::Allocator,
    instance_buffer: &mut BufferAllocation<vk::AccelerationStructureInstanceKHR>,
    primitive_blas: &Vec<vk::AccelerationStructureKHR>,
    instance: &SceneInstance,
) {
    let blas = primitive_blas[instance.primitive_index];
    let blas_address = unsafe {
        context
            .acceleration_structure_pfn
            .get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(blas),
            )
    };

    unsafe {
        instance_buffer.append_unsafe(
            &[vk::AccelerationStructureInstanceKHR {
                transform: glam_to_khr(instance.world_from_local),

                // closesthit.slang reads this via InstanceIndex()
                instance_custom_index_and_mask: Packed24_8::new(
                    instance.primitive_index as u32,
                    0xFF,
                ),

                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::empty().as_raw() as _,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_address,
                },
            }],
            allocator,
        );
    }
}

fn build_blas(
    context: &lazy_vulkan::Context,
    allocator: &mut lazy_vulkan::Allocator,
    command_buffer: vk::CommandBuffer,
    primitive: &ScenePrimitive,
) -> vk::AccelerationStructureKHR {
    let primitive_count = primitive.index_count / 3;
    let geometries = &[vk::AccelerationStructureGeometryKHR::default()
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: primitive.indices,
                })
                .index_type(vk::IndexType::UINT32)
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: primitive.vertices,
                })
                .vertex_stride(std::mem::size_of::<lazy_vulkan_gltf::Vertex>() as _)
                .max_vertex(primitive.vertex_count - 1),
        })
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)];

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        context
            .acceleration_structure_pfn
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(geometries),
                &[primitive_count],
                &mut size_info,
            )
    };

    // Build a buffer for the BLAS storage. This is essentially opaque storage used by the driver
    let blas_storage = allocator.allocate_buffer::<u8>(
        size_info.acceleration_structure_size as _,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
    );

    let blas = unsafe {
        context
            .acceleration_structure_pfn
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .buffer(blas_storage.handle)
                    .size(size_info.acceleration_structure_size),
                None,
            )
    }
    .unwrap();

    let scratch_buffer = allocator.allocate_buffer_with_alignment::<u8>(
        size_info.build_scratch_size as _,
        context
            .raytracing_properties
            .min_acceleration_structure_scratch_offset_alignment as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    );

    unsafe {
        context
            .acceleration_structure_pfn
            .cmd_build_acceleration_structures(
                command_buffer,
                &[vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(geometries)
                    .dst_acceleration_structure(blas)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer.device_address,
                    })],
                &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(primitive_count)]],
            )
    };
    blas
}

fn build_tlas(
    context: &lazy_vulkan::Context,
    allocator: &mut lazy_vulkan::Allocator,
    scene_data: &mut SceneData,
    command_buffer: vk::CommandBuffer,
) -> vk::AccelerationStructureKHR {
    //
    // We start with instance geometries, which are the instances of the BLASes.
    let instance_count = scene_data.instance_buffer.len() as u32;
    let instance_geometries = &[vk::AccelerationStructureGeometryKHR::default()
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: scene_data.instance_buffer.device_address,
                }),
        })
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)];

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        context
            .acceleration_structure_pfn
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(instance_geometries),
                &[instance_count],
                &mut size_info,
            )
    };

    // Allocate some storage for the TLAS. This is essentially opaque storage used by the driver
    let tlas_storage = allocator.allocate_buffer::<u8>(
        size_info.acceleration_structure_size as _,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
    );

    let tlas = unsafe {
        context
            .acceleration_structure_pfn
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .buffer(tlas_storage.handle)
                    .size(size_info.acceleration_structure_size),
                None,
            )
    }
    .unwrap();

    // Now allocate a scratch buffer for the build operation
    let scratch_buffer = allocator.allocate_buffer_with_alignment::<u8>(
        size_info.build_scratch_size as _,
        context
            .raytracing_properties
            .min_acceleration_structure_scratch_offset_alignment as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    );

    // Now build the TLAS using the BLAS geometries and the scratch buffer
    unsafe {
        context
            .acceleration_structure_pfn
            .cmd_build_acceleration_structures(
                command_buffer,
                &[vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(instance_geometries)
                    .dst_acceleration_structure(tlas)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer.device_address,
                    })],
                &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(instance_count)]],
            )
    };
    tlas
}

pub fn glam_to_khr(transform: glam::Affine3A) -> vk::TransformMatrixKHR {
    let cols = transform.to_cols_array_2d();

    // needs to be row major
    vk::TransformMatrixKHR {
        matrix: [
            cols[0][0], cols[1][0], cols[2][0], cols[3][0], // row 0
            cols[0][1], cols[1][1], cols[2][1], cols[3][1], // row 1
            cols[0][2], cols[1][2], cols[2][2], cols[3][2], // row 2
        ],
    }
}

struct SBT {
    gen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    call_region: vk::StridedDeviceAddressRegionKHR,
}

impl SBT {
    fn new(
        context: &lazy_vulkan::Context,
        allocator: &mut lazy_vulkan::Allocator,
        pipeline: vk::Pipeline,
    ) -> SBT {
        let miss_count = 2;
        let hit_count = 1;
        let handle_count = 1 + miss_count + hit_count;
        let raytracing_properties = &context.raytracing_properties;
        let handle_size = raytracing_properties.shader_group_handle_size;
        let handle_size_aligned = align_up_pow2(
            handle_size as _,
            raytracing_properties.shader_group_handle_alignment as _,
        );

        let stride = align_up_pow2(
            handle_size_aligned,
            raytracing_properties.shader_group_base_alignment as _,
        );
        let mut gen_region = vk::StridedDeviceAddressRegionKHR::default()
            .stride(stride)
            .size(stride);

        let mut miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .stride(handle_size_aligned)
            .size(align_up_pow2(
                miss_count * handle_size_aligned,
                raytracing_properties.shader_group_base_alignment as u64,
            ));

        let mut hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .stride(handle_size_aligned)
            .size(align_up_pow2(
                hit_count * handle_size_aligned,
                raytracing_properties.shader_group_base_alignment as u64,
            ));

        let call_region = vk::StridedDeviceAddressRegionKHR::default();

        let data_size = handle_count * (handle_size as u64);

        let handles = unsafe {
            context
                .ray_tracing_pipeline_pfn
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    handle_count as u32,
                    data_size as usize,
                )
        }
        .unwrap();

        // Copy data
        let sbt_size = gen_region.size + miss_region.size + hit_region.size + call_region.size;
        let mut sbt_data = vec![0; sbt_size as usize];

        let handle_size_usize = handle_size as usize;

        let mut copy_group = |dst_offset: usize, group_index: usize| {
            let src_offset = group_index * handle_size_usize;

            sbt_data[dst_offset..dst_offset + handle_size_usize]
                .copy_from_slice(&handles[src_offset..src_offset + handle_size_usize]);
        };

        copy_group(0, 0);

        let miss_offset = gen_region.size as usize;
        copy_group(miss_offset, 1);
        copy_group(miss_offset + miss_region.stride as usize, 2);

        let hit_offset = miss_offset + miss_region.size as usize;
        copy_group(hit_offset, 3);

        let mut sbt_buffer = allocator.allocate_buffer::<u8>(
            sbt_size as usize,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        );

        sbt_buffer.append(&sbt_data, allocator);

        gen_region.device_address = sbt_buffer.device_address;
        miss_region.device_address = gen_region.device_address + gen_region.size;
        hit_region.device_address = gen_region.device_address + gen_region.size + miss_region.size;

        SBT {
            gen_region,
            miss_region,
            hit_region,
            call_region,
        }
    }
}

// TLAS helpers

// SBT helpers

pub const fn align_up_pow2(value: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    (value + (alignment - 1)) & !(alignment - 1)
}

pub const fn is_pow2(a: u64) -> bool {
    a != 0 && (a & (a - 1)) == 0
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Primitive {
    material: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,
    vertex_buffer: vk::DeviceAddress,
}

unsafe impl bytemuck::Zeroable for Primitive {}
unsafe impl bytemuck::Pod for Primitive {}

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {
    view_inverse: glam::Mat4,
    proj_inverse: glam::Mat4,
    primitive_buffer: vk::DeviceAddress,
    tunnel_bay_length_metres: f32,
    frame: u32,
    light_buffer: vk::DeviceAddress,
    light_count: u32,
    train_track_s_metres: f32,
    track_length_metres: f32,
}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

fn r(path: &str) -> Vec<u8> {
    std::fs::read(path).context(path.to_string()).unwrap()
}

#[repr(C)]
#[derive(Copy, Clone)]
struct TonemappingRegisters {
    texture_id: u32,
}

unsafe impl bytemuck::Zeroable for TonemappingRegisters {}
unsafe impl bytemuck::Pod for TonemappingRegisters {}

/// Camera helpers
fn camera_view_from_train_frame(train_frame: TrackFrame, demo_state: &DemoState) -> glam::Mat4 {
    let speed_fraction = (demo_state.speed_mps / (150.0 / 3.6)).clamp(0.0, 1.0);
    let t = demo_state.elapsed_s;

    // A bit of cab-space motion
    let sway_x_m = (t * 1.7).sin() * speed_fraction * 0.065;
    let bob_y_m = (t * 4.2).sin() * speed_fraction * 0.022;
    let nod_y_m = (t * 3.1).sin() * speed_fraction * 0.045;

    let camera_in_track = glam::vec3(sway_x_m, 1.55 + bob_y_m + nod_y_m, -1.5);
    let look_target_in_track = glam::vec3(sway_x_m * 0.35, 1.35 + bob_y_m + nod_y_m, 30.0);

    let track_offset_to_world = |offset_m: glam::Vec3| {
        train_frame.origin
            + train_frame.right * offset_m.x
            + train_frame.up * offset_m.y
            + train_frame.forward * offset_m.z
    };

    let eye_world = track_offset_to_world(camera_in_track);

    let target_world = track_offset_to_world(look_target_in_track);

    let camera_forward = (target_world - eye_world).normalize();
    let camera_right = train_frame.up.cross(camera_forward).normalize();
    let camera_up = camera_forward.cross(camera_right).normalize();

    // This maps raygen camera space to world space:
    // local +X -> train/image right
    // local +Y -> train/image up
    // local -Z -> train forward
    let world_from_camera = glam::Mat4::from_cols(
        camera_right.extend(0.0),
        camera_up.extend(0.0),
        (-camera_forward).extend(0.0),
        eye_world.extend(1.0),
    );

    world_from_camera
}

// Pipeline helpers

fn create_rt_pipeline(
    pipeline_layout: vk::PipelineLayout,
    context: &Arc<lazy_vulkan::Context>,
) -> vk::Pipeline {
    let pipeline = unsafe {
        context
            .ray_tracing_pipeline_pfn
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[vk::RayTracingPipelineCreateInfoKHR::default()
                    .groups(&[
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .general_shader(RAYGEN_INDEX),
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .general_shader(MISS_INDEX),
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .general_shader(SHADOW_MISS_INDEX),
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(CLOSEST_HIT_INDEX)
                            .intersection_shader(vk::SHADER_UNUSED_KHR)
                            .general_shader(vk::SHADER_UNUSED_KHR),
                    ])
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                            .name(c"main")
                            .module(lazy_vulkan::load_module(&r(RAYGEN_SHADER_PATH), context)),
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::MISS_KHR)
                            .name(c"main")
                            .module(lazy_vulkan::load_module(&r(MISS_SHADER_PATH), context)),
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::MISS_KHR)
                            .name(c"main")
                            .module(lazy_vulkan::load_module(
                                &r(SHADOW_MISS_SHADER_PATH),
                                context,
                            )),
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .name(c"main")
                            .module(lazy_vulkan::load_module(
                                &r(CLOSEST_HIT_SHADER_PATH),
                                context,
                            )),
                    ])
                    .max_pipeline_ray_recursion_depth(2)
                    .layout(pipeline_layout)],
                None,
            )
    }
    .unwrap()[0];
    pipeline
}

// Lights
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TunnelLight {
    position: glam::Vec3,
    radius_metres: f32,
    colour: glam::Vec3,
    intensity: f32,
    start_s_metres: f32,
}

unsafe impl bytemuck::Zeroable for TunnelLight {}
unsafe impl bytemuck::Pod for TunnelLight {}

fn generate_tunnel_lights(
    track: &Track,
    led_fixtures: &[mesh_generation::LedTubeFixture],
    lower_strip_fixtures: &[mesh_generation::LowerStripFixture],
) -> Vec<TunnelLight> {
    let mut lights = Vec::with_capacity(led_fixtures.len() + lower_strip_fixtures.len());

    lights.extend(led_fixtures.iter().map(|fixture| {
        let center_s_metres = fixture.start_s_metres + fixture.length_metres * 0.5;
        let frame = track.sample(center_s_metres);

        let position = frame.origin + frame.right * fixture.x_metres + frame.up * fixture.y_metres;

        TunnelLight {
            position,
            radius_metres: fixture.radius_metres,
            colour: fixture.colour,
            intensity: fixture.intensity,
            start_s_metres: fixture.start_s_metres,
        }
    }));

    for fixture in lower_strip_fixtures {
        for tile in mesh_generation::lower_strip_tiles(fixture) {
            let center_s_metres = (tile.start_s_metres + tile.end_s_metres) * 0.5;
            let frame = track.sample(center_s_metres);

            let inward_face_x_metres = fixture.x_metres - fixture.half_width_metres;

            let position =
                frame.origin + frame.right * inward_face_x_metres + frame.up * fixture.y_metres;

            lights.push(TunnelLight {
                position,
                radius_metres: fixture.radius_metres,
                colour: fixture.colour,
                intensity: fixture.intensity,
                start_s_metres: tile.start_s_metres,
            });
        }
    }

    lights
}
