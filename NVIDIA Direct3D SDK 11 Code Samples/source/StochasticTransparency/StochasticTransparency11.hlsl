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

Texture2D<uint>    tRandoms             : register(t0);
Texture2DMS<float> tStochasticDepth     : register(t0);
Texture2D<float3>  tBackgroundColor     : register(t0);
Texture2D<float>   tTotalAlphaBuffer    : register(t1);
Texture2D<float4>  tAccumulationBuffer  : register(t2);

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

//Total Alpha Pass
float TotalAlphaPS ( Geometry_VSOut IN ) : SV_Target
{
    float alpha = ShadeFragment(IN.Normal).a;
    return alpha;
}

//Stochastic Depth Pass
uint StochasticDepthPS ( Geometry_VSOut IN, uint PrimitiveID : SV_PrimitiveID ) : SV_Coverage
{
	//AlphaToCoverage Is Uncorrelated!

    float alpha = ShadeFragment(IN.Normal).a;
    return randmaskwide(uint2(IN.HPosition.xy), alpha, PrimitiveID);
}

//Total Alpha Pass And Accumulation Pass Can Merge ???

//Accumulation Pass
float4 AccumulationPS ( Geometry_VSOut IN ) : SV_Target
{
	//Estimate Visibility
	//3.2 Stochastic Shadow Maps
	//svis(z) = count(z<=zi)/S ≈ vis(z) //考虑Reverse-Z

	int2 pos2d = int2(IN.HPosition.xy); //Sample 8 Times While Shading Once //We Cast From float To int
	float z = IN.HPosition.z;
	uint count = 0;
	
	[unroll]
	for (uint sampleId = 0; sampleId < NUM_MSAA_SAMPLES; ++sampleId)
	{
		float zi = tStochasticDepth.Load(pos2d, sampleId).r;
		if (z <= zi)
		{
			++count;
		}
	}

	float visz = ((float)count) / ((float)NUM_MSAA_SAMPLES);

	//3.4 Depth-Based Stochastic Transparency
	//C = Σ vis(z)*a*c
    float4 c = ShadeFragment(IN.Normal);

	//4.2 Bias of Depth-Based Methods
	//U = Σ visz * c * a
	//U1 = Σ visz * a //The "R/S"
	float ac = visz * c.a;
	return float4(ac * c.rgb, ac);
}

float4 CompositePS(FullscreenVSOut IN) : SV_Target
{
    int2 pos2d = int2(IN.pos.xy);

	float3 backgroundColor = tBackgroundColor.Load(int3(pos2d, 0)).rgb;
	float transmittance = tTotalAlphaBuffer.Load(int3(pos2d, 0)).r;

	//4.2 Bias of Depth-Based Methods
#if USE_ALPHA_CORRECTION
	float4 t_U_U1 = tAccumulationBuffer.Load(int3(pos2d, 0));
	float3 U = t_U_U1.rgb; //U = visz * c * a
	float U1 = t_U_U1.a; //U1 = visz * a //The "R/S"

	float AC = 1.0 - transmittance; //TotalAlpha
	float3 D = (U1 > 0.0) ? (AC*U / U1) : 0.0;

	float3 transparentColor = D;
#else
	 //Too Dark
	float3 transparentColor = tAccumulationBuffer.Load(int3(pos2d, 0)).rgb;
#endif

	//Under Operator
	return float4(transparentColor + transmittance * backgroundColor, 1.0);
}