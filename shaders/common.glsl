struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
};

layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};
layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint indices[];
};

layout(buffer_reference, scalar) readonly buffer Material {
    vec4 baseColourFactor;
    vec3 emissiveColourFactor;
    uint baseColourTextureID;
    uint normalTextureID;
    uint metallicRoughnessTextureID;
    uint aoTextureID;
};

struct Primitive {
    Material material;
    IndexBuffer indexBuffer; // Points to the base of this primitive
    VertexBuffer vertexBuffer; // Points to the base of this primitive
};

layout(buffer_reference, scalar) readonly buffer PrimitiveBuffer {
    Primitive primitives[];
};

layout(push_constant) uniform Registers {
    mat4 viewInverse;
    mat4 projInverse;
    PrimitiveBuffer primitiveBuffer;
    uint frame;
} registers;

struct HitPayload {
    vec3 hitValue;
    uint seed;
    uint depth;
    vec3 rayOrigin;
    vec3 rayDirection;
    vec3 weight;
};
