use std::path::{Path, PathBuf};

use lazy_vulkan::{Renderer, SlabUpload, StateFamily, vk};
use lazy_vulkan_gltf::{GPUMaterial, TextureID};

pub fn load_material<SF: StateFamily>(
    renderer: &mut Renderer<SF>,
    name: impl AsRef<str>,
) -> SlabUpload<GPUMaterial> {
    let material_dir = Path::new("assets").join(name.as_ref());

    let base_colour = load_texture(
        renderer,
        &material_dir,
        "base_colour.png",
        vk::Format::R8G8B8A8_SRGB,
    );
    let normal = load_texture(
        renderer,
        &material_dir,
        "normal.png",
        vk::Format::R8G8B8A8_UNORM,
    );
    let orm = load_texture(
        renderer,
        &material_dir,
        "orm.png",
        vk::Format::R8G8B8A8_UNORM,
    );

    let material = GPUMaterial {
        base_colour_factor: glam::Vec4::ONE,
        emissive_colour_factor: glam::Vec3::ZERO,
        base_colour_texture: base_colour,
        normal_texture: normal,
        metallic_roughness_texture: orm,
        ao_texture: orm,
    };

    renderer.allocator.upload_to_slab(&[material])
}

fn load_texture<SF: StateFamily>(
    renderer: &mut Renderer<SF>,
    material_dir: &Path,
    file_name: &str,
    format: vk::Format,
) -> TextureID {
    let path = required_texture_path(material_dir, file_name);
    let image = renderer.create_sampled_image_from_png(
        format!("{} {}", material_dir.display(), file_name),
        path,
        format,
    );

    image.id.into()
}

fn required_texture_path(material_dir: &Path, file_name: &str) -> PathBuf {
    let path = material_dir.join(file_name);
    assert!(
        path.exists(),
        "missing material texture: {}",
        path.display()
    );
    path
}
