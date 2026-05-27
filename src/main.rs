use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::Context;
use lazy_vulkan::{
    BufferAllocation, FULL_IMAGE, LayerInfo, LazyVulkan, PipelineOptions, StateFamily, SubRenderer,
    ash::vk::{self, Packed24_8},
};
use winit::{application::ApplicationHandler, window::WindowAttributes};

static CLOSEST_SHADER_PATH: &'static str = "shaders/closesthit.rchit.spv";
static MISS_SHADER_PATH: &'static str = "shaders/miss.rmiss.spv";
static RAYGEN_SHADER_PATH: &'static str = "shaders/raygen.rgen.spv";
static TONEMAPPING_SHADER_PATH: &'static str = "shaders/tonemapping.frag.spv";
static FULLSCREEN_SHADER_PATH: &'static str = "shaders/fullscreen.vert.spv";

const CORRIDOR_REPEAT_COUNT: usize = 9;
const CORRIDOR_SPACING_METRES: f32 = 11.0;
const CAMERA_SPEED_METRES_PER_SECOND: f32 = 2.5;

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {
    view_inverse: glam::Mat4,
    proj_inverse: glam::Mat4,
    primitive_buffer: vk::DeviceAddress,
    frame: u32,
}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

#[repr(C)]
#[derive(Copy, Clone)]
struct TonemappingRegisters {
    texture_id: u32,
}

unsafe impl bytemuck::Zeroable for TonemappingRegisters {}
unsafe impl bytemuck::Pod for TonemappingRegisters {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Primitive {
    material: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,
    vertex_buffer: vk::DeviceAddress,
}

unsafe impl bytemuck::Zeroable for Primitive {}
unsafe impl bytemuck::Pod for Primitive {}

pub struct RTRenderer {
    context: Arc<lazy_vulkan::Context>,
    image: lazy_vulkan::Image,
    state: Option<RTState>,
    vertex_buffer: BufferAllocation<lazy_vulkan_gltf::Vertex>,
    index_buffer: BufferAllocation<u32>,
    primitive_buffer: BufferAllocation<Primitive>,
    instance_buffer: BufferAllocation<vk::AccelerationStructureInstanceKHR>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    tonemapping_pipeline: lazy_vulkan::Pipeline,
    tonemapping_descriptor_set: vk::DescriptorSet,
    #[allow(unused)]
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    asset: lazy_vulkan_gltf::LoadedAsset,
}

pub struct RTState {
    #[allow(unused)]
    tlas: vk::AccelerationStructureKHR,
    gen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    call_region: vk::StridedDeviceAddressRegionKHR,
}

impl RTRenderer {
    pub fn new(renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>) -> Self {
        let extent = renderer.get_drawable_extent();
        let image = renderer.create_image(
            "RT Target",
            vk::Format::R8G8B8A8_UNORM,
            extent,
            &[],
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        );

        let mut vertex_buffer = renderer.allocator.allocate_buffer(
            10 * 1024 * 1024,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let mut index_buffer = renderer.allocator.allocate_buffer(
            1024 * 1024,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let mut primitive_buffer = renderer
            .allocator
            .allocate_buffer(10 * 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER);

        let asset = lazy_vulkan_gltf::load_asset(
            "test_assets/cornellBox.gltf",
            &mut renderer.allocator,
            &mut renderer.image_manager,
            &mut index_buffer,
            &mut vertex_buffer,
        )
        .unwrap();

        let mut source_instance_count = 0;

        let mut primitives = HashMap::new();
        for node in &asset.nodes {
            let mesh = &asset.meshes[usize::from(node.mesh_id)];
            source_instance_count += mesh.primitives.len();

            for primitive in &mesh.primitives {
                let index_buffer = index_buffer.device_address + primitive.index_buffer_offset;
                let vertex_buffer = vertex_buffer.device_address + primitive.vertex_buffer_offset;

                let primitive_id = primitive.id;
                let primitive = Primitive {
                    material: primitive.material,
                    index_buffer,
                    vertex_buffer,
                };

                primitives.insert(primitive_id, primitive);
            }
        }

        // Generate a corridor of instances
        let repeated_instance_count = source_instance_count * CORRIDOR_REPEAT_COUNT;

        let mut primitive_data = Vec::new();
        for _ in 0..CORRIDOR_REPEAT_COUNT {
            for node in &asset.nodes {
                let mesh = &asset.meshes[usize::from(node.mesh_id)];
                for primitive in &mesh.primitives {
                    primitive_data.push(*primitives.get(&primitive.id).unwrap());
                }
            }
        }

        log::debug!("Primitive data: {primitive_data:?}");
        primitive_buffer.append(&primitive_data, &mut renderer.allocator);

        // Create the instance buffer
        let instance_buffer = renderer
            .allocator
            .allocate_buffer::<vk::AccelerationStructureInstanceKHR>(
                repeated_instance_count,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            );

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

        let raygen_index = 0;
        let miss_index = 1;
        let closest_hit_index = 2;

        let context = &renderer.context;

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
                                .general_shader(raygen_index),
                            vk::RayTracingShaderGroupCreateInfoKHR::default()
                                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                                .intersection_shader(vk::SHADER_UNUSED_KHR)
                                .general_shader(miss_index),
                            vk::RayTracingShaderGroupCreateInfoKHR::default()
                                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                                .closest_hit_shader(closest_hit_index)
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
                                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                                .name(c"main")
                                .module(lazy_vulkan::load_module(&r(CLOSEST_SHADER_PATH), context)),
                        ])
                        .layout(pipeline_layout)],
                    None,
                )
        }
        .unwrap()[0];

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
            vertex_buffer,
            index_buffer,
            primitive_buffer,
            instance_buffer,
            descriptor_pool,
            descriptor_set,
            pipeline,
            pipeline_layout,
            tonemapping_pipeline,
            tonemapping_descriptor_set: renderer.descriptors.set,
            asset,
        }
    }

    fn create_rt_state(&mut self, allocator: &mut lazy_vulkan::Allocator) -> RTState {
        let command_buffer = self.context.draw_command_buffer;
        let vertex_buffer = &mut self.vertex_buffer;

        // (TODO): This is kinda yuck.
        //
        // Solutions:
        // 1) `append_now(command_buffer)` <-- does what it says on the tin
        // 2) `on_transferred` <-- add a functor to be called when the transfer has been recorded
        // 3) `Buffer.flush(command_buffer` / `allocator.flush_buffer(buffer, command_buffer)` <-- executes just the transfers for this buffer
        allocator.execute_transfers(command_buffer);

        let mut blas_map = HashMap::new();
        for mesh in &self.asset.meshes {
            for primitive in &mesh.primitives {
                let primitive_count = primitive.index_count / 3;
                let geometries = &[vk::AccelerationStructureGeometryKHR::default()
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: self.index_buffer.device_address
                                    + primitive.index_buffer_offset,
                            })
                            .index_type(vk::IndexType::UINT32)
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: vertex_buffer.device_address
                                    + primitive.vertex_buffer_offset,
                            })
                            .vertex_stride(std::mem::size_of::<lazy_vulkan_gltf::Vertex>() as _)
                            .max_vertex(primitive.vertex_count - 1),
                    })
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)];

                let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(geometries);

                let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
                unsafe {
                    self.context
                        .acceleration_structure_pfn
                        .get_acceleration_structure_build_sizes(
                            vk::AccelerationStructureBuildTypeKHR::DEVICE,
                            &build_geometry_info,
                            &[primitive_count],
                            &mut size_info,
                        )
                };

                // This is essentially opaque storage used by the driver
                let blas_storage = allocator.allocate_buffer::<u8>(
                    size_info.acceleration_structure_size as _,
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                );

                let blas = unsafe {
                    self.context
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
                    self.context
                        .raytracing_properties
                        .min_acceleration_structure_scratch_offset_alignment
                        as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                );

                let build_geometry_info = build_geometry_info
                    .dst_acceleration_structure(blas)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer.device_address,
                    });

                unsafe {
                    self.context
                        .acceleration_structure_pfn
                        .cmd_build_acceleration_structures(
                            command_buffer,
                            &[build_geometry_info],
                            &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                                .primitive_count(primitive_count)]],
                        )
                };

                let key = format!("{}{}", mesh.id, primitive.id);
                blas_map.insert(key, blas);
            }
        }

        for corridor_transform in
            generate_corridor_instance_transforms(CORRIDOR_REPEAT_COUNT, CORRIDOR_SPACING_METRES)
        {
            for node in &self.asset.nodes {
                let mesh = &self.asset.meshes[usize::from(node.mesh_id)];
                for primitive in &mesh.primitives {
                    let key = format!("{}{}", mesh.id, primitive.id);
                    let Some(blas) = blas_map.get(&key).copied() else {
                        continue;
                    };

                    let blas_address = unsafe {
                        self.context
                            .acceleration_structure_pfn
                            .get_acceleration_structure_device_address(
                                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                    .acceleration_structure(blas),
                            )
                    };

                    unsafe {
                        self.instance_buffer.append_unsafe(
                            &[vk::AccelerationStructureInstanceKHR {
                                transform: glam_to_khr(corridor_transform * node.transform),
                                instance_custom_index_and_mask: Packed24_8::new(
                                    primitive.id.into(),
                                    0xFF,
                                ),
                                instance_shader_binding_table_record_offset_and_flags:
                                    Packed24_8::new(
                                        0,
                                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                                            .as_raw() as _,
                                    ),
                                acceleration_structure_reference:
                                    vk::AccelerationStructureReferenceKHR {
                                        device_handle: blas_address,
                                    },
                            }],
                            allocator,
                        )
                    };
                }
            }
        }

        allocator.execute_transfers(command_buffer);
        unsafe {
            self.context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(self.instance_buffer.handle)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .size(vk::WHOLE_SIZE),
                ]),
            )
        };

        let instance_count = self.instance_buffer.len() as u32;

        let instance_geometries = &[vk::AccelerationStructureGeometryKHR::default()
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.instance_buffer.device_address,
                    }),
            })
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)];

        let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(instance_geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.context
                .acceleration_structure_pfn
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geometry_info,
                    &[instance_count],
                    &mut size_info,
                )
        };

        // This is essentially opaque storage used by the driver
        let tlas_storage = allocator.allocate_buffer::<u8>(
            size_info.acceleration_structure_size as _,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        );

        let tlas = unsafe {
            self.context
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

        // This is essentially opaque storage used by the driver
        let scratch_buffer = allocator.allocate_buffer_with_alignment::<u8>(
            size_info.build_scratch_size as _,
            self.context
                .raytracing_properties
                .min_acceleration_structure_scratch_offset_alignment as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let build_geometry_info = build_geometry_info
            .dst_acceleration_structure(tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address,
            });

        unsafe {
            self.context
                .acceleration_structure_pfn
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[build_geometry_info],
                    &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(instance_count)]],
                )
        };

        // SBT!
        let miss_count = 1;
        let hit_count = 1;
        let handle_count = 1 + miss_count + hit_count;
        let raytracing_properties = &self.context.raytracing_properties;
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
            self.context
                .ray_tracing_pipeline_pfn
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    handle_count as u32,
                    data_size as usize,
                )
        }
        .unwrap();

        // Copy data
        let sbt_size = gen_region.size + miss_region.size + hit_region.size + call_region.size;
        let mut sbt_data = vec![0; sbt_size as usize];

        let mut offset = 0;
        sbt_data[offset..handle_size as usize].copy_from_slice(&handles[..handle_size as usize]);
        offset += gen_region.size as usize;
        sbt_data[offset..offset + handle_size as usize]
            .copy_from_slice(&handles[handle_size as usize..(handle_size as usize) * 2]);

        offset += miss_region.size as usize;
        sbt_data[offset..offset + handle_size as usize]
            .copy_from_slice(&handles[(handle_size as usize * 2)..(handle_size as usize) * 3]);

        let mut sbt_buffer = allocator.allocate_buffer::<u8>(
            sbt_size as usize,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        );

        sbt_buffer.append(&sbt_data, allocator);

        gen_region.device_address = sbt_buffer.device_address;
        miss_region.device_address = gen_region.device_address + gen_region.size;
        hit_region.device_address = gen_region.device_address + gen_region.size + miss_region.size;

        allocator.execute_transfers(command_buffer);

        let context = &self.context;
        let device = &context.device;

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
                                .acceleration_structures(&[tlas]),
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

        let rtstate = RTState {
            tlas,
            gen_region,
            miss_region,
            hit_region,
            call_region,
        };
        rtstate
    }
}

fn r(path: &str) -> Vec<u8> {
    std::fs::read(path).context(path.to_string()).unwrap()
}

impl<'a> SubRenderer<'a> for RTRenderer {
    type State = RenderState;

    fn stage_transfers(
        &mut self,
        _state: &Self::State,
        allocator: &mut lazy_vulkan::Allocator,
        _image_manager: &mut lazy_vulkan::ImageManager,
    ) {
        if self.state.is_some() {
            return;
        }

        self.state = Some(self.create_rt_state(allocator));
    }

    fn draw_layer(
        &mut self,
        state: &Self::State,
        context: &lazy_vulkan::Context,
        layer_info: LayerInfo,
    ) {
        let Some(rt_state) = &self.state else { return };

        let device = &context.device;
        let command_buffer = context.draw_command_buffer;
        let drawable = &layer_info.colour_attachment.unwrap();

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
            let mut perspective =
                glam::Mat4::perspective_rh(60_f32.to_radians(), aspect_ratio, 0.01, 10000.0);

            // wulkankjzk
            perspective.y_axis *= -1.0;

            let corridor_half_length =
                (CORRIDOR_REPEAT_COUNT as f32 - 1.0) * CORRIDOR_SPACING_METRES * 0.5;
            let camera_x = (state.elapsed_seconds * CAMERA_SPEED_METRES_PER_SECOND * 0.1).sin()
                * corridor_half_length;

            let eye = glam::vec3(camera_x, 1.5, 15.0);
            let target = glam::vec3(camera_x + 2.5, 1.0, 0.0);

            let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

            let registers = Registers {
                view_inverse: view.inverse(),
                proj_inverse: perspective.inverse(),
                primitive_buffer: self.primitive_buffer.device_address,
                frame: 0, // TODO
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
                &rt_state.gen_region,
                &rt_state.miss_region,
                &rt_state.hit_region,
                &rt_state.call_region,
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

            // context.device.cmd_blit_image(
            //     command_buffer,
            //     self.image.handle,
            //     vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            //     drawable.image,
            //     vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            //     &[vk::ImageBlit::default()
            //         .src_offsets([
            //             vk::Offset3D::default(),
            //             vk::Offset3D::default()
            //                 .x(drawable.extent.width as i32)
            //                 .y(drawable.extent.height as i32)
            //                 .z(1),
            //         ])
            //         .src_subresource(
            //             vk::ImageSubresourceLayers::default()
            //                 .aspect_mask(vk::ImageAspectFlags::COLOR)
            //                 .layer_count(1),
            //         )
            //         .dst_offsets([
            //             vk::Offset3D::default(),
            //             vk::Offset3D::default()
            //                 .x(drawable.extent.width as i32)
            //                 .y(drawable.extent.height as i32)
            //                 .z(1),
            //         ])
            //         .dst_subresource(
            //             vk::ImageSubresourceLayers::default()
            //                 .aspect_mask(vk::ImageAspectFlags::COLOR)
            //                 .layer_count(1),
            //         )],
            //     vk::Filter::LINEAR,
            // );
        };
    }

    fn label(&self) -> &'static str {
        "RT Renderer"
    }
}

fn generate_corridor_instance_transforms(count: usize, spacing_meters: f32) -> Vec<glam::Affine3A> {
    let center = (count as f32 - 1.0) * 0.5;

    (0..count)
        .map(|index| {
            let x_offset = (index as f32 - center) * spacing_meters;
            glam::Affine3A::from_translation(glam::vec3(x_offset, 0.0, 0.0))
        })
        .collect()
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

pub const fn align_up_pow2(value: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    (value + (alignment - 1)) & !(alignment - 1)
}

pub const fn is_pow2(a: u64) -> bool {
    a != 0 && (a & (a - 1)) == 0
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Hello RT"))
            .unwrap();

        let mut lazy_vulkan = LazyVulkan::from_window(&window);
        let renderer = RTRenderer::new(&mut lazy_vulkan.renderer);
        lazy_vulkan.add_sub_renderer(Box::new(renderer));

        self.state = Some(State {
            window,
            lazy_vulkan,
            start_time: Instant::now(),
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
                state.lazy_vulkan.draw(&RenderState {
                    elapsed_seconds: state.start_time.elapsed().as_secs_f32(),
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

pub struct RenderState {
    elapsed_seconds: f32,
}

pub struct RenderStateFamily;

impl StateFamily for RenderStateFamily {
    type For<'a_> = RenderState;
}

struct State {
    #[allow(unused)]
    window: winit::window::Window,
    lazy_vulkan: LazyVulkan<RenderStateFamily>,
    start_time: Instant,
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

fn compile_shaders() {
    const SHADERS: &[(&str, &str, &str)] = &[
        (
            "shaders/raygen.slang",
            "raygeneration",
            "shaders/raygen.rgen.spv",
        ),
        ("shaders/miss.slang", "miss", "shaders/miss.rmiss.spv"),
        (
            "shaders/closesthit.slang",
            "closesthit",
            "shaders/closesthit.rchit.spv",
        ),
        (
            "shaders/fullscreen.slang",
            "vertex",
            "shaders/fullscreen.vert.spv",
        ),
        (
            "shaders/tonemapping.slang",
            "fragment",
            "shaders/tonemapping.frag.spv",
        ),
    ];

    for (input_path, stage, output_path) in SHADERS {
        log::debug!("[SHADERS] Compiling {input_path} to {output_path}");

        let status = std::process::Command::new("slangc")
            .arg(format!("./{}", input_path))
            .arg("-entry")
            .arg("main")
            .arg("-stage")
            .arg(stage)
            .arg("-target")
            .arg("spirv")
            .arg("-profile")
            .arg("glsl_460")
            .arg("-fvk-use-entrypoint-name")
            .arg("-matrix-layout-column-major")
            .arg("-fvk-use-scalar-layout")
            .arg("-g")
            .arg("-o")
            .arg(output_path)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        if !status.success() {
            panic!("Failed to compile shader {input_path}");
        }
    }
}

fn main() {
    env_logger::init();
    compile_shaders();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
