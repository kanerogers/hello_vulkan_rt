import common;

[shader("miss")]
void main(inout HitPayload payload) {
  if (payload.depth == 0) {
    payload.hitValue = float3(0.0);
  } else {
    payload.hitValue = float3(0.001);
  }

  payload.depth = 100;
}
