// Copyright (c) 2011 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, NONINFRINGEMENT,IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA 
// OR ITS SUPPLIERS BE  LIABLE  FOR  ANY  DIRECT, SPECIAL,  INCIDENTAL,  INDIRECT,  OR  
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS 
// OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY 
// OTHER PECUNIARY LOSS) ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, 
// EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
// Please direct any bugs or questions to SDKFeedback@nvidia.com

#include "BaseTechnique.hlsl"

// Assuming that tStochasticDepth  have 8 samples per pixel
#define NUM_MSAA_SAMPLES 8

// Alpha correction (Section 3.2 of the paper)
// Removes noise for pixels with low depth complexity (1-2 layers / pixel)
#define USE_ALPHA_CORRECTION 1

Texture2D<uint>    tRandoms                                      : register(t0);
Texture2DMS<float> tStochasticDepth                              : register(t0);
Texture2D<float3>  tBackgroundColor                              : register(t0);
Texture2D<float4>  tStochasticColorAndCorrectTotalAlphaBuffer    : register(t1);
Texture2D<float>   tStochasticTotalAlphaBuffer                   : register(t2);

// from http://www.concentric.net/~Ttwang/tech/inthash.htm
uint ihash(uint seed)
{
    seed = (seed+0x7ed55d16u) + (seed<<12);
    seed = (seed^0xc761c23cu) ^ (seed>>19);
    seed = (seed+0x165667b1u) + (seed<<5);
    seed = (seed+0xd3a2646cu) ^ (seed<<9);
    seed = (seed+0xfd7046c5u) + (seed<<3);
    seed = (seed^0xb55a4f09u) ^ (seed>>16);
    return seed;
}

// pixelPos = 2D screen-space position for the current fragment
// primID = auto-generated primitive id for the current fragment
uint getlayerseed(uint2 pixelPos, int primID)
{
    // Seeding by primitive id, as described in Section 5 of the paper.
    uint layerseed = primID * 32;

    // For simulating more than 8 samples per pixel, the algorithm
    // can be run in multiple passes with different random offsets.
    layerseed += g_randomOffset;

    layerseed += (pixelPos.x << 10) + (pixelPos.y << 20);
    return layerseed;
}

// Compute mask based on primitive id and alpha
uint randmaskwide(uint2 pixelPos, float alpha, int primID)
{
    // Generate seed based on fragment coords and primitive id
    uint seed = getlayerseed(pixelPos, primID);

    // Compute a hash with uniform distribution
    seed = ihash(seed);

    // Modulo operation
    seed &= g_randMaskSizePowOf2MinusOne;

    // Quantize alpha from [0.0,1.0] to [0,ALPHA_VALUES]
    uint2 coords = uint2(seed, alpha * g_randMaskAlphaValues);

    return tRandoms.Load(int3(coords, 0)).r;
}

//Stochastic Depth Pass
uint StochasticDepthPS ( Geometry_VSOut IN, uint PrimitiveID : SV_PrimitiveID ) : SV_Coverage
{
	//AlphaToCoverage Is Uncorrelated!

    float alpha = ShadeFragment(IN.Normal).a;
    return randmaskwide(uint2(IN.HPosition.xy), alpha, PrimitiveID);
}

//We Merge TotalAlpha Pass And Accumulation Pass Together.
struct Pixel_PSOut
{
	float4 StochasticColorAndCorrectTotalAlpha : SV_Target0;
	float4 StochasticTotalAlpha                : SV_Target1;
};

//TotalAlpha And Accumulate Pass
Pixel_PSOut TotalAlphaAndAccumulatePS( Geometry_VSOut IN )
{
	//Estimate Visibility
	//3.2 Stochastic Shadow Maps
	//svis(z) = count(z<=zi)/S ≈ vis(z) //考虑Reverse-Z

	int2 pos2d = int2(IN.HPosition.xy); //Sample 8 Times While Shading Once //We Cast From float To int
	
	//浮点数转换成定点数后再进行大小比较

	//Step1：浮点数->定点数“原码”
	//floatBitsToInt/asint

	//Step2：定点数“原码”->定点数“补码”
	//符号位为0：不变
	//符号位为1：符号位不变且其余各位按位取反（相当于按位异或0X7FFFFFFF）->在以上结果的基础上加1
	//X -> ((X & 0X80000000) != 0) ? ((X ^ 0X7FFFFFFF) + 1) : X

	float z = IN.HPosition.z;
	int z_Trueform = asint(z); //浮点数->定点数“原码”
	int z_Complement = ((z_Trueform & 0x80000000) != 0) ? ((z_Trueform ^ 0x7FFFFFFF) + 1) : z_Trueform; //定点数“原码”->定点数“补码”
	//z_Complement = z_Complement & 0xFFFFFC00; //Depth Bias 忽略最后10位

	//Try ???
	//SampleCmp SamplerComparisonState ???

	uint count = 0;
	//[unroll]
	for (uint sampleId = 0; sampleId < NUM_MSAA_SAMPLES; ++sampleId)
	{
		float zi = tStochasticDepth.Load(pos2d, sampleId).r;
		int zi_Trueform = asint(zi); //浮点数->定点数“原码”
		int zi_Complement = ((zi_Trueform & 0x80000000) != 0) ? ((zi_Trueform ^ 0x7FFFFFFF) + 1) : zi_Trueform; //定点数“原码”->定点数“补码”
		//zi_Complement = zi_Complement & 0xFFFFFC00;
		if (z_Complement <= (zi_Complement + 0x400/*Depth Bias*/)) //Depth Bias 忽略最后10位
		{
			++count;
		}
	}

	float visz = ((float)count) / ((float)NUM_MSAA_SAMPLES);

	//3.4 Depth-Based Stochastic Transparency
	//C = Σ vis(z)*a*c
    float4 rgba = ShadeFragment(IN.Normal);
	float3 c = rgba.rgb;
	float a = rgba.a;

	//4.2 Bias of Depth-Based Methods
	//U = Σ visz * c * a
	//U1 = Σ visz * a //The "R/S"
	float ac = visz * a;

	Pixel_PSOut rtval;
	rtval.StochasticColorAndCorrectTotalAlpha = float4(ac * c, a);
	rtval.StochasticTotalAlpha = float4(ac, ac, ac, ac);
	return rtval;
}

float4 CompositePS(FullscreenVSOut IN) : SV_Target
{
    int2 pos2d = int2(IN.pos.xy);

	//Step 1: Total Alpha Correction
#if USE_ALPHA_CORRECTION
	//4.2 Bias of Depth-Based Methods
	float4 UAndT = tStochasticColorAndCorrectTotalAlphaBuffer.Load(int3(pos2d, 0));
	float3 U = UAndT.rgb; //U = visz * c * a
	float transmittance = UAndT.a; //TotalAlpha  
	
	float U1 = tStochasticTotalAlphaBuffer.Load(int3(pos2d, 0)).r; //U1 = visz * a //The "R/S"

	float AC = 1.0 - transmittance; //TotalAlpha
	float3 D = (U1 > 0.0) ? (AC*U / U1) : 0.0;

	float3 transparentColor = D;
#else
	 //Too Dark
	float4 UAndT = tStochasticColorAndCorrectTotalAlphaBuffer.Load(int3(pos2d, 0));
	float3 transparentColor = UAndT.rgb; //U = visz * c * a
	float transmittance = UAndT.a; //TotalAlpha  
#endif

	//Step 2: Under Operator
	float3 backgroundColor = tBackgroundColor.Load(int3(pos2d, 0)).rgb;
	return float4(transparentColor + transmittance * backgroundColor, 1.0);
}


struct Basic_Pixel_PSOut
{
	float4 StochasticColorAndAlpha : SV_Target0;
	uint AlphaToCoverage           : SV_Coverage;
	float4 CorrectTotalAlpha       : SV_Target1;
};

Basic_Pixel_PSOut BasicStochasticTransparencyPS(Geometry_VSOut IN, uint PrimitiveID : SV_PrimitiveID)
{
	float4 rgba = ShadeFragment(IN.Normal);
	float a = rgba.a;

	Basic_Pixel_PSOut rtval;
	rtval.StochasticColorAndAlpha = rgba;
	rtval.AlphaToCoverage = randmaskwide(uint2(IN.HPosition.xy), a, PrimitiveID);
	//rtval.CorrectTotalAlpha = float4(a, a, a, a);
	return rtval;
}

#if 0

Texture2D<float3>    tBackgroundColor                            : register(t0);
Texture2DMS<float4>  tStochasticColorAndAlphaBuffer              : register(t1);
//Texture2D<float>     CorrectTotalAlpha                           : register(t2);

float4 CompositePS(FullscreenVSOut IN) : SV_Target
{
	int2 pos2d = int2(IN.pos.xy);

	//Step 1: Total Alpha Correction
#if USE_ALPHA_CORRECTION
	//4.2 Bias of Depth-Based Methods
	float4 UAndT = tStochasticColorAndCorrectTotalAlphaBuffer.Load(int3(pos2d, 0));
	float3 U = UAndT.rgb; //U = visz * c * a
	float transmittance = UAndT.a; //TotalAlpha  

	float U1 = tStochasticTotalAlphaBuffer.Load(int3(pos2d, 0)).r; //U1 = visz * a //The "R/S"

	float AC = 1.0 - transmittance; //TotalAlpha
	float3 D = (U1 > 0.0) ? (AC*U / U1) : 0.0;

	float3 transparentColor = D;
#else
	 //Too Dark
	float4 UAndT = tStochasticColorAndCorrectTotalAlphaBuffer.Load(int3(pos2d, 0));
	float3 transparentColor = UAndT.rgb; //U = visz * c * a
	float transmittance = UAndT.a; //TotalAlpha  
#endif

	//Step 2: Under Operator
	float3 backgroundColor = tBackgroundColor.Load(int3(pos2d, 0)).rgb;
	return float4(transparentColor + transmittance * backgroundColor, 1.0);
}

#endif