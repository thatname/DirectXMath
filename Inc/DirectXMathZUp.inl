//-------------------------------------------------------------------------------------
// DirectXMathMatrix.inl -- SIMD C++ Math library
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615560
//-------------------------------------------------------------------------------------

#pragma once

/*
	Functions reimplemented in Z-up coordinate system.
	X-Front-Roll
	Y-Right-Pitch
	Z-Up-Yaw
*/

inline XMMATRIX XM_CALLCONV XMMatrixLookAt
(
	FXMVECTOR EyePosition,
	FXMVECTOR FocusPosition,
	FXMVECTOR UpDirection
) noexcept
{
	XMVECTOR EyeDirection = XMVectorSubtract(FocusPosition, EyePosition);
	return XMMatrixLookTo(EyePosition, EyeDirection, UpDirection);
}

//------------------------------------------------------------------------------

inline XMMATRIX XM_CALLCONV XMMatrixLookTo
(
	FXMVECTOR EyePosition,
	FXMVECTOR EyeDirection,
	FXMVECTOR UpDirection
) noexcept
{
	assert(!XMVector3Equal(EyeDirection, XMVectorZero()));
	assert(!XMVector3IsInfinite(EyeDirection));
	assert(!XMVector3Equal(UpDirection, XMVectorZero()));
	assert(!XMVector3IsInfinite(UpDirection));

	XMVECTOR R2 = XMVector3Normalize(EyeDirection);

	XMVECTOR R0 = XMVector3Cross(UpDirection, R2);
	R0 = XMVector3Normalize(R0);

	XMVECTOR R1 = XMVector3Cross(R2, R0);

	XMVECTOR NegEyePosition = XMVectorNegate(EyePosition);

	XMVECTOR D0 = XMVector3Dot(R0, NegEyePosition);
	XMVECTOR D1 = XMVector3Dot(R1, NegEyePosition);
	XMVECTOR D2 = XMVector3Dot(R2, NegEyePosition);

	XMMATRIX M;
	M.r[1] = XMVectorSelect(D0, R0, g_XMSelect1110.v);
	M.r[2] = XMVectorSelect(D1, R1, g_XMSelect1110.v);
	M.r[0] = XMVectorSelect(D2, R2, g_XMSelect1110.v);
	M.r[3] = g_XMIdentityR3.v;
	
	M = XMMatrixTranspose(M);

	return M;
}

inline XMMATRIX XM_CALLCONV XMMatrixPerspective
(
	float ViewWidth,
	float ViewHeight,
	float NearZ,
	float FarZ
) noexcept
{
	assert(NearZ > 0.f && FarZ > 0.f);
	assert(!XMScalarNearEqual(ViewWidth, 0.0f, 0.00001f));
	assert(!XMScalarNearEqual(ViewHeight, 0.0f, 0.00001f));
	assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

#if defined(_XM_NO_INTRINSICS_)

	float TwoNearZ = NearZ + NearZ;
	float fRange = FarZ / (FarZ - NearZ);

	XMMATRIX M;
	M.m[1][0] = TwoNearZ / ViewWidth;
	M.m[1][1] = 0.0f;
	M.m[1][2] = 0.0f;
	M.m[1][3] = 0.0f;

	M.m[2][0] = 0.0f;
	M.m[2][1] = TwoNearZ / ViewHeight;
	M.m[2][2] = 0.0f;
	M.m[2][3] = 0.0f;

	M.m[0][0] = 0.0f;
	M.m[0][1] = 0.0f;
	M.m[0][2] = fRange;
	M.m[0][3] = 1.0f;

	M.m[3][0] = 0.0f;
	M.m[3][1] = 0.0f;
	M.m[3][2] = -fRange * NearZ;
	M.m[3][3] = 0.0f;
	return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	float TwoNearZ = NearZ + NearZ;
	float fRange = FarZ / (FarZ - NearZ);
	const XMVECTOR Zero = vdupq_n_f32(0);
	XMMATRIX M;
	M.r[1] = vsetq_lane_f32(TwoNearZ / ViewWidth, Zero, 0);
	M.r[2] = vsetq_lane_f32(TwoNearZ / ViewHeight, Zero, 1);
	M.r[0] = vsetq_lane_f32(fRange, g_XMIdentityR3.v, 2);
	M.r[3] = vsetq_lane_f32(-fRange * NearZ, Zero, 2);
	return M;
#elif defined(_XM_SSE_INTRINSICS_)
	XMMATRIX M;
	float TwoNearZ = NearZ + NearZ;
	float fRange = FarZ / (FarZ - NearZ);
	// Note: This is recorded on the stack
	XMVECTOR rMem = {
		TwoNearZ / ViewWidth,
		TwoNearZ / ViewHeight,
		fRange,
		-fRange * NearZ
	};
	// Copy from memory to SSE register
	XMVECTOR vValues = rMem;
	XMVECTOR vTemp = _mm_setzero_ps();
	// Copy x only
	vTemp = _mm_move_ss(vTemp, vValues);
	// TwoNearZ / ViewWidth,0,0,0
	M.r[1] = vTemp;
	// 0,TwoNearZ / ViewHeight,0,0
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskY);
	M.r[2] = vTemp;
	// x=fRange,y=-fRange * NearZ,0,1.0f
	vValues = _mm_shuffle_ps(vValues, g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
	// 0,0,fRange,1.0f
	vTemp = _mm_setzero_ps();
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
	M.r[0] = vTemp;
	// 0,0,-fRange * NearZ,0
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
	M.r[3] = vTemp;
	return M;
#endif
}

inline XMMATRIX XM_CALLCONV XMMatrixPerspectiveFov
(
	float FovAngleY,
	float AspectRatio,
	float NearZ,
	float FarZ
) noexcept
{
	assert(NearZ > 0.f && FarZ > 0.f);
	assert(!XMScalarNearEqual(FovAngleY, 0.0f, 0.00001f * 2.0f));
	assert(!XMScalarNearEqual(AspectRatio, 0.0f, 0.00001f));
	assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

#if defined(_XM_NO_INTRINSICS_)

	float    SinFov;
	float    CosFov;
	XMScalarSinCos(&SinFov, &CosFov, 0.5f * FovAngleY);

	float Height = CosFov / SinFov;
	float Width = Height / AspectRatio;
	float fRange = FarZ / (FarZ - NearZ);

	XMMATRIX M;
	M.m[1][0] = Width;
	M.m[1][1] = 0.0f;
	M.m[1][2] = 0.0f;
	M.m[1][3] = 0.0f;

	M.m[2][0] = 0.0f;
	M.m[2][1] = Height;
	M.m[2][2] = 0.0f;
	M.m[2][3] = 0.0f;

	M.m[0][0] = 0.0f;
	M.m[0][1] = 0.0f;
	M.m[0][2] = fRange;
	M.m[0][3] = 1.0f;

	M.m[3][0] = 0.0f;
	M.m[3][1] = 0.0f;
	M.m[3][2] = -fRange * NearZ;
	M.m[3][3] = 0.0f;
	return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	float    SinFov;
	float    CosFov;
	XMScalarSinCos(&SinFov, &CosFov, 0.5f * FovAngleY);

	float fRange = FarZ / (FarZ - NearZ);
	float Height = CosFov / SinFov;
	float Width = Height / AspectRatio;
	const XMVECTOR Zero = vdupq_n_f32(0);

	XMMATRIX M;
	M.r[1] = vsetq_lane_f32(Width, Zero, 0);
	M.r[2] = vsetq_lane_f32(Height, Zero, 1);
	M.r[0] = vsetq_lane_f32(fRange, g_XMIdentityR3.v, 2);
	M.r[3] = vsetq_lane_f32(-fRange * NearZ, Zero, 2);
	return M;
#elif defined(_XM_SSE_INTRINSICS_)
	float    SinFov;
	float    CosFov;
	XMScalarSinCos(&SinFov, &CosFov, 0.5f * FovAngleY);

	float fRange = FarZ / (FarZ - NearZ);
	// Note: This is recorded on the stack
	float Height = CosFov / SinFov;
	XMVECTOR rMem = {
		Height / AspectRatio,
		Height,
		fRange,
		-fRange * NearZ
	};
	// Copy from memory to SSE register
	XMVECTOR vValues = rMem;
	XMVECTOR vTemp = _mm_setzero_ps();
	// Copy x only
	vTemp = _mm_move_ss(vTemp, vValues);
	// CosFov / SinFov,0,0,0
	XMMATRIX M;
	M.r[1] = vTemp;
	// 0,Height / AspectRatio,0,0
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskY);
	M.r[2] = vTemp;
	// x=fRange,y=-fRange * NearZ,0,1.0f
	vTemp = _mm_setzero_ps();
	vValues = _mm_shuffle_ps(vValues, g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
	// 0,0,fRange,1.0f
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
	M.r[0] = vTemp;
	// 0,0,-fRange * NearZ,0.0f
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
	M.r[3] = vTemp;
	return M;
#endif
}

inline XMMATRIX XM_CALLCONV XMMatrixPerspectiveOffCenter
(
	float ViewLeft,
	float ViewRight,
	float ViewBottom,
	float ViewTop,
	float NearZ,
	float FarZ
) noexcept
{
	assert(NearZ > 0.f && FarZ > 0.f);
	assert(!XMScalarNearEqual(ViewRight, ViewLeft, 0.00001f));
	assert(!XMScalarNearEqual(ViewTop, ViewBottom, 0.00001f));
	assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

#if defined(_XM_NO_INTRINSICS_)

	float TwoNearZ = NearZ + NearZ;
	float ReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float ReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = FarZ / (FarZ - NearZ);

	XMMATRIX M;
	M.m[1][0] = TwoNearZ * ReciprocalWidth;
	M.m[1][1] = 0.0f;
	M.m[1][2] = 0.0f;
	M.m[1][3] = 0.0f;

	M.m[2][0] = 0.0f;
	M.m[2][1] = TwoNearZ * ReciprocalHeight;
	M.m[2][2] = 0.0f;
	M.m[2][3] = 0.0f;

	M.m[0][0] = -(ViewLeft + ViewRight) * ReciprocalWidth;
	M.m[0][1] = -(ViewTop + ViewBottom) * ReciprocalHeight;
	M.m[0][2] = fRange;
	M.m[0][3] = 1.0f;

	M.m[3][0] = 0.0f;
	M.m[3][1] = 0.0f;
	M.m[3][2] = -fRange * NearZ;
	M.m[3][3] = 0.0f;
	return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	float TwoNearZ = NearZ + NearZ;
	float ReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float ReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = FarZ / (FarZ - NearZ);
	const XMVECTOR Zero = vdupq_n_f32(0);

	XMMATRIX M;
	M.r[1] = vsetq_lane_f32(TwoNearZ * ReciprocalWidth, Zero, 0);
	M.r[2] = vsetq_lane_f32(TwoNearZ * ReciprocalHeight, Zero, 1);
	M.r[0] = XMVectorSet(-(ViewLeft + ViewRight) * ReciprocalWidth,
		-(ViewTop + ViewBottom) * ReciprocalHeight,
		fRange,
		1.0f);
	M.r[3] = vsetq_lane_f32(-fRange * NearZ, Zero, 2);
	return M;
#elif defined(_XM_SSE_INTRINSICS_)
	XMMATRIX M;
	float TwoNearZ = NearZ + NearZ;
	float ReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float ReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = FarZ / (FarZ - NearZ);
	// Note: This is recorded on the stack
	XMVECTOR rMem = {
		TwoNearZ * ReciprocalWidth,
		TwoNearZ * ReciprocalHeight,
		-fRange * NearZ,
		0
	};
	// Copy from memory to SSE register
	XMVECTOR vValues = rMem;
	XMVECTOR vTemp = _mm_setzero_ps();
	// Copy x only
	vTemp = _mm_move_ss(vTemp, vValues);
	// TwoNearZ*ReciprocalWidth,0,0,0
	M.r[1] = vTemp;
	// 0,TwoNearZ*ReciprocalHeight,0,0
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskY);
	M.r[2] = vTemp;
	// 0,0,fRange,1.0f
	M.r[0] = XMVectorSet(-(ViewLeft + ViewRight) * ReciprocalWidth,
		-(ViewTop + ViewBottom) * ReciprocalHeight,
		fRange,
		1.0f);
	// 0,0,-fRange * NearZ,0.0f
	vValues = _mm_and_ps(vValues, g_XMMaskZ);
	M.r[3] = vValues;
	return M;
#endif
}

inline XMMATRIX XM_CALLCONV XMMatrixOrthographic
(
	float ViewWidth,
	float ViewHeight,
	float NearZ,
	float FarZ
) noexcept
{
	assert(!XMScalarNearEqual(ViewWidth, 0.0f, 0.00001f));
	assert(!XMScalarNearEqual(ViewHeight, 0.0f, 0.00001f));
	assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

#if defined(_XM_NO_INTRINSICS_)

	float fRange = 1.0f / (FarZ - NearZ);

	XMMATRIX M;
	M.m[1][0] = 2.0f / ViewWidth;
	M.m[1][1] = 0.0f;
	M.m[1][2] = 0.0f;
	M.m[1][3] = 0.0f;

	M.m[2][0] = 0.0f;
	M.m[2][1] = 2.0f / ViewHeight;
	M.m[2][2] = 0.0f;
	M.m[2][3] = 0.0f;

	M.m[0][0] = 0.0f;
	M.m[0][1] = 0.0f;
	M.m[0][2] = fRange;
	M.m[0][3] = 0.0f;

	M.m[3][0] = 0.0f;
	M.m[3][1] = 0.0f;
	M.m[3][2] = -fRange * NearZ;
	M.m[3][3] = 1.0f;
	return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	float fRange = 1.0f / (FarZ - NearZ);

	const XMVECTOR Zero = vdupq_n_f32(0);
	XMMATRIX M;
	M.r[1] = vsetq_lane_f32(2.0f / ViewWidth, Zero, 0);
	M.r[2] = vsetq_lane_f32(2.0f / ViewHeight, Zero, 1);
	M.r[0] = vsetq_lane_f32(fRange, Zero, 2);
	M.r[3] = vsetq_lane_f32(-fRange * NearZ, g_XMIdentityR3.v, 2);
	return M;
#elif defined(_XM_SSE_INTRINSICS_)
	XMMATRIX M;
	float fRange = 1.0f / (FarZ - NearZ);
	// Note: This is recorded on the stack
	XMVECTOR rMem = {
		2.0f / ViewWidth,
		2.0f / ViewHeight,
		fRange,
		-fRange * NearZ
	};
	// Copy from memory to SSE register
	XMVECTOR vValues = rMem;
	XMVECTOR vTemp = _mm_setzero_ps();
	// Copy x only
	vTemp = _mm_move_ss(vTemp, vValues);
	// 2.0f / ViewWidth,0,0,0
	M.r[1] = vTemp;
	// 0,2.0f / ViewHeight,0,0
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskY);
	M.r[2] = vTemp;
	// x=fRange,y=-fRange * NearZ,0,1.0f
	vTemp = _mm_setzero_ps();
	vValues = _mm_shuffle_ps(vValues, g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
	// 0,0,fRange,0.0f
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 0, 0, 0));
	M.r[0] = vTemp;
	// 0,0,-fRange * NearZ,1.0f
	vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 1, 0, 0));
	M.r[3] = vTemp;
	return M;
#endif
}

inline XMMATRIX XM_CALLCONV XMMatrixOrthographicOffCenter
(
	float ViewLeft,
	float ViewRight,
	float ViewBottom,
	float ViewTop,
	float NearZ,
	float FarZ
) noexcept
{
	assert(!XMScalarNearEqual(ViewRight, ViewLeft, 0.00001f));
	assert(!XMScalarNearEqual(ViewTop, ViewBottom, 0.00001f));
	assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

#if defined(_XM_NO_INTRINSICS_)

	float ReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float ReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = 1.0f / (FarZ - NearZ);

	XMMATRIX M;
	M.m[1][0] = ReciprocalWidth + ReciprocalWidth;
	M.m[1][1] = 0.0f;
	M.m[1][2] = 0.0f;
	M.m[1][3] = 0.0f;

	M.m[2][0] = 0.0f;
	M.m[2][1] = ReciprocalHeight + ReciprocalHeight;
	M.m[2][2] = 0.0f;
	M.m[2][3] = 0.0f;

	M.m[0][0] = 0.0f;
	M.m[0][1] = 0.0f;
	M.m[0][2] = fRange;
	M.m[0][3] = 0.0f;

	M.m[3][0] = -(ViewLeft + ViewRight) * ReciprocalWidth;
	M.m[3][1] = -(ViewTop + ViewBottom) * ReciprocalHeight;
	M.m[3][2] = -fRange * NearZ;
	M.m[3][3] = 1.0f;
	return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	float ReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float ReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = 1.0f / (FarZ - NearZ);
	const XMVECTOR Zero = vdupq_n_f32(0);
	XMMATRIX M;
	M.r[1] = vsetq_lane_f32(ReciprocalWidth + ReciprocalWidth, Zero, 0);
	M.r[2] = vsetq_lane_f32(ReciprocalHeight + ReciprocalHeight, Zero, 1);
	M.r[0] = vsetq_lane_f32(fRange, Zero, 2);
	M.r[3] = XMVectorSet(-(ViewLeft + ViewRight) * ReciprocalWidth,
		-(ViewTop + ViewBottom) * ReciprocalHeight,
		-fRange * NearZ,
		1.0f);
	return M;
#elif defined(_XM_SSE_INTRINSICS_)
	XMMATRIX M;
	float fReciprocalWidth = 1.0f / (ViewRight - ViewLeft);
	float fReciprocalHeight = 1.0f / (ViewTop - ViewBottom);
	float fRange = 1.0f / (FarZ - NearZ);
	// Note: This is recorded on the stack
	XMVECTOR rMem = {
		fReciprocalWidth,
		fReciprocalHeight,
		fRange,
		1.0f
	};
	XMVECTOR rMem2 = {
		-(ViewLeft + ViewRight),
		-(ViewTop + ViewBottom),
		-NearZ,
		1.0f
	};
	// Copy from memory to SSE register
	XMVECTOR vValues = rMem;
	XMVECTOR vTemp = _mm_setzero_ps();
	// Copy x only
	vTemp = _mm_move_ss(vTemp, vValues);
	// fReciprocalWidth*2,0,0,0
	vTemp = _mm_add_ss(vTemp, vTemp);
	M.r[1] = vTemp;
	// 0,fReciprocalHeight*2,0,0
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskY);
	vTemp = _mm_add_ps(vTemp, vTemp);
	M.r[2] = vTemp;
	// 0,0,fRange,0.0f
	vTemp = vValues;
	vTemp = _mm_and_ps(vTemp, g_XMMaskZ);
	M.r[0] = vTemp;
	// -(ViewLeft + ViewRight)*fReciprocalWidth,-(ViewTop + ViewBottom)*fReciprocalHeight,fRange*-NearZ,1.0f
	vValues = _mm_mul_ps(vValues, rMem2);
	M.r[3] = vValues;
	return M;
#endif
}

inline XMMATRIX XM_CALLCONV XMMatrixRotationRollPitchYaw
(
	float Roll,
	float Pitch,
	float Yaw
) noexcept
{
	XMVECTOR Angles = XMVectorSet(Roll, Pitch, Yaw, 0.0f);
	return XMMatrixRotationRollPitchYawFromVector(Angles);
}

//------------------------------------------------------------------------------

inline XMMATRIX XM_CALLCONV XMMatrixRotationRollPitchYawFromVector
(
	FXMVECTOR Angles // <Pitch, Yaw, Roll, undefined>
) noexcept
{
	XMVECTOR Q = XMQuaternionRotationRollPitchYawFromVector(Angles);
	return XMMatrixRotationQuaternion(Q);
}

inline XMVECTOR XM_CALLCONV XMQuaternionRotationRollPitchYaw
(
	float Pitch,
	float Yaw,
	float Roll
) noexcept
{
	XMVECTOR Angles = XMVectorSet(Pitch, Yaw, Roll, 0.0f);
	XMVECTOR Q = XMQuaternionRotationRollPitchYawFromVector(Angles);
	return Q;
}

//------------------------------------------------------------------------------

inline XMVECTOR XM_CALLCONV XMQuaternionRotationRollPitchYawFromVector
(
	FXMVECTOR Angles // <Pitch, Yaw, Roll, 0>
) noexcept
{
	static const XMVECTORF32  Sign = { { { 1.0f, -1.0f, -1.0f, 1.0f } } };

	XMVECTOR HalfAngles = XMVectorMultiply(Angles, g_XMOneHalf.v);

	XMVECTOR SinAngles, CosAngles;
	XMVectorSinCos(&SinAngles, &CosAngles, HalfAngles);

	XMVECTOR R0 = XMVectorPermute<XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X>(SinAngles, CosAngles);
	XMVECTOR P0 = XMVectorPermute<XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y>(SinAngles, CosAngles);
	XMVECTOR Y0 = XMVectorPermute<XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z>(SinAngles, CosAngles);
	XMVECTOR R1 = XMVectorPermute<XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X>(CosAngles, SinAngles);
	XMVECTOR P1 = XMVectorPermute<XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y>(CosAngles, SinAngles);
	XMVECTOR Y1 = XMVectorPermute<XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z>(CosAngles, SinAngles);

	XMVECTOR Q1 = XMVectorMultiply(P1, Sign.v);
	XMVECTOR Q0 = XMVectorMultiply(P0, Y0);
	Q1 = XMVectorMultiply(Q1, Y1);
	Q0 = XMVectorMultiply(Q0, R0);
	XMVECTOR Q = XMVectorMultiplyAdd(Q1, R1, Q0);

	return Q;
}