import common;

[shader("closesthit")]
void main(inout ShadowPayload payload,
          BuiltInTriangleIntersectionAttributes attributes) {
  payload.hit = true;
}
