#version 460 core
#extension GL_EXT_ray_tracing : require

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    // Barycentric color
    payload = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}
