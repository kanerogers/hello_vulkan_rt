import common;

[shader("miss")]
void main(inout ShadowPayload payload) { payload.hit = false; }
