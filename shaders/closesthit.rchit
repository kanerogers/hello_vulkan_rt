#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#include "common.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    Primitive primitive = registers.primitiveBuffer.primitives[gl_InstanceCustomIndexEXT];
    IndexBuffer i = primitive.indexBuffer;
    ivec3 triangleIndex = ivec3(i.indices[0], i.indices[1], i.indices[2]);

    // Barycentrics
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Vertices
    const Vertex v0 = primitive.vertexBuffer.vertices[triangleIndex.x];
    const Vertex v1 = primitive.vertexBuffer.vertices[triangleIndex.y];
    const Vertex v2 = primitive.vertexBuffer.vertices[triangleIndex.z];

    const vec3 pos0 = v0.position;
    const vec3 pos1 = v1.position;
    const vec3 pos2 = v2.position;

    const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
    payload = position;
}
