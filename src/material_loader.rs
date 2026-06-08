use std::{
    fs::File,
    path::{Path, PathBuf},
};

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
    let orm = load_orm_texture(renderer, &material_dir);

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

fn load_orm_texture<SF: StateFamily>(
    renderer: &mut Renderer<SF>,
    material_dir: &Path,
) -> TextureID {
    let orm_path = material_dir.join("orm.png");
    if orm_path.exists() {
        return load_texture(
            renderer,
            material_dir,
            "orm.png",
            vk::Format::R8G8B8A8_UNORM,
        );
    }

    let ao = decode_png_channel(&required_texture_path(material_dir, "ao.png"));
    let roughness = decode_png_channel(&required_texture_path(material_dir, "roughness.png"));
    let metalness = decode_png_channel(&required_texture_path(material_dir, "metalness.png"));

    assert_eq!(
        ao.extent,
        roughness.extent,
        "ao.png and roughness.png must have matching dimensions in {}",
        material_dir.display()
    );
    assert_eq!(
        ao.extent,
        metalness.extent,
        "ao.png and metalness.png must have matching dimensions in {}",
        material_dir.display()
    );

    let pixel_count = (ao.extent.width * ao.extent.height) as usize;
    let mut orm_pixels = Vec::with_capacity(pixel_count * 4);

    for pixel_index in 0..pixel_count {
        orm_pixels.push(ao.values[pixel_index]);
        orm_pixels.push(roughness.values[pixel_index]);
        orm_pixels.push(metalness.values[pixel_index]);
        orm_pixels.push(255);
    }

    let image = renderer.create_image(
        format!("{} packed orm", material_dir.display()),
        vk::Format::R8G8B8A8_UNORM,
        ao.extent,
        orm_pixels,
        vk::ImageUsageFlags::SAMPLED,
    );

    image.id.into()
}

struct DecodedChannel {
    extent: vk::Extent2D,
    values: Vec<u8>,
}

fn decode_png_channel(path: &Path) -> DecodedChannel {
    let file = File::open(path).unwrap_or_else(|error| {
        panic!("failed to open texture {}: {error}", path.display());
    });

    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info().unwrap_or_else(|error| {
        panic!("failed to read PNG info for {}: {error}", path.display());
    });

    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).unwrap_or_else(|error| {
        panic!("failed to decode PNG {}: {error}", path.display());
    });

    let pixels = &pixels[..info.buffer_size()];
    let channel_count = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::Rgb => 3,
        png::ColorType::Indexed => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgba => 4,
    };

    let values = pixels
        .chunks_exact(channel_count)
        .map(|channels| channels[0])
        .collect::<Vec<_>>();

    DecodedChannel {
        extent: vk::Extent2D {
            width: info.width,
            height: info.height,
        },
        values,
    }
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
