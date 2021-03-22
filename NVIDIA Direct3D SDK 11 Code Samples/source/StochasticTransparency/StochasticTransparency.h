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

#pragma once
#include "SimpleRT.h"
#include "BaseTechnique.h"
#include "Scene.h"
#include "MersenneTwister.h"
#include <algorithm>

#include "StochasticTransparency_StochasticDepthPS.h"
#include "StochasticTransparency_AccumulateAndTotalAlphaPS.h"
#include "StochasticTransparency_CompositePS.h"

#define RANDOM_SIZE 2048
#define ALPHA_VALUES 256
#define NUM_MSAA_SAMPLES 8
#define MAX_NUM_PASSES 8

//The AccumulationBuffer may not be MSAA
#define STOCHASTIC_COLOR_FORMAT DXGI_FORMAT_R8G8B8A8_UNORM

// Consists of a depth-stencil buffer //Texture2D //D24_UNORM_S8_UINT or D32_FLOAT
// and the associated depth-stencil view for binding.
class StochasticDepth
{
public:
	ID3D11Texture2D *pTexture;
	ID3D11DepthStencilView *pDSV;
	ID3D11ShaderResourceView *pSRV;

	StochasticDepth(ID3D11Device* pd3dDevice, UINT Width, UINT Height)
		: pTexture(NULL)
		, pDSV(NULL)
		, pSRV(NULL)
	{
		HRESULT hr;
		D3D11_TEXTURE2D_DESC texDesc;
		texDesc.ArraySize = 1;
		texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
		texDesc.CPUAccessFlags = NULL;
		texDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		texDesc.Width = Width;
		texDesc.Height = Height;
		texDesc.MipLevels = 1;
		texDesc.MiscFlags = NULL;
		texDesc.SampleDesc.Count = NUM_MSAA_SAMPLES;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Usage = D3D11_USAGE_DEFAULT;
		V(pd3dDevice->CreateTexture2D(&texDesc, NULL, &pTexture));
		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
		dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
		dsvDesc.Flags = 0U;
		//dsvDesc.Texture2DMS.UnusedField_NothingToDefine
		V(pd3dDevice->CreateDepthStencilView(pTexture, &dsvDesc, &pDSV));
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
		//srvDesc.Texture2DMS.UnusedField_NothingToDefine
		V(pd3dDevice->CreateShaderResourceView(pTexture, &srvDesc, &pSRV));
	}

	~StochasticDepth()
	{
		SAFE_RELEASE(pTexture);
		SAFE_RELEASE(pDSV);
		SAFE_RELEASE(pSRV);
	}
};

class StochasticTransparency : public BaseTechnique, public Scene
{
public:
    StochasticTransparency(ID3D11Device* pd3dDevice, UINT Width, UINT Height)
        : BaseTechnique(pd3dDevice)
        , m_pBackgroundRenderTarget(NULL)
        , m_pBackgroundDepth(NULL)
		, m_pStochasticColorAndCorrectTotalAlphaRenderTarget(NULL)
		, m_pStochasticTotalAlphaRenderTarget(NULL)
		, m_pStochasticDepth(NULL)
		, m_pStochasticDepthPS(NULL)
		, m_pTotalAlphaAndAccumulatePS(NULL)
		, m_pCompositePS(NULL)
        , m_pRndTexture(NULL)
        , m_pRndTextureSRV(NULL)
        , m_pTotalAlphaAndAccumulateBS(NULL)
    {
		CreateFrameBuffer(pd3dDevice, Width, Height);
		CreateStochasticDepth(pd3dDevice, Width, Height);
        CreateRandomBitmasks(pd3dDevice);
        CreateBlendStates(pd3dDevice);
        CreateShaders(pd3dDevice);
        CBData.randMaskSizePowOf2MinusOne = RANDOM_SIZE - 1;
        CBData.randMaskAlphaValues = ALPHA_VALUES;
    }

    virtual void Render(ID3D11DeviceContext* pd3dImmediateContext, ID3D11RenderTargetView *pBackBuffer)
    {
		ID3DUserDefinedAnnotation *pPerf;
		pd3dImmediateContext->QueryInterface(IID_PPV_ARGS(&pPerf));

		//The BackgroudColor may not be MSAA
		
		//----------------------------------------------------------------------------------
		// 1. Render Opaque Background
		//----------------------------------------------------------------------------------
        //The background colors should be initialized by drawing the opaque objects in the scene.
		pPerf->BeginEvent(L"Opaque Pass");
		float ClearColorBack[4] = { m_BackgroundColor.x, m_BackgroundColor.y, m_BackgroundColor.z, 0 };
		pd3dImmediateContext->ClearRenderTargetView(m_pBackgroundRenderTarget->pRTV, ClearColorBack);
		float ClearDepthBack = 1.0f;
		pd3dImmediateContext->ClearDepthStencilView(m_pBackgroundDepth->pDSV, D3D11_CLEAR_DEPTH, ClearDepthBack, 0U);
		pPerf->EndEvent();

        // Update the constant buffer
        pd3dImmediateContext->UpdateSubresource(m_pParamsCB, 0, NULL, &CBData, 0, 0);

        // Set shared states
        pd3dImmediateContext->IASetInputLayout(m_pInputLayout);
        pd3dImmediateContext->VSSetShader(m_pGeometryVS, NULL, 0);
        pd3dImmediateContext->GSSetShader(NULL, NULL, 0);
        pd3dImmediateContext->RSSetState(m_pNoCullRS);
        pd3dImmediateContext->VSSetConstantBuffers(0, 1, &m_pParamsCB);
        pd3dImmediateContext->PSSetConstantBuffers(0, 1, &m_pParamsCB);
        pd3dImmediateContext->PSSetConstantBuffers(1, 1, &m_pShadingParamsCB);

		//By the limit of the hardware, the maximum sample count of MSAA is 8X MSAA.
		//The author proposed that we can use multiple passes to simulate more sample counts.
		//Due to the performance issue, we only use one pass.
        //for (UINT LayerId = 0; LayerId < m_NumPasses; ++LayerId)
        {
			CBData.randomOffset = 0;
            pd3dImmediateContext->UpdateSubresource(m_pParamsCB, 0, NULL, &CBData, 0, 0);

            //----------------------------------------------------------------------------------
            // 2. Render MSAA "stochastic depths", writting SV_Coverage in the pixel shader
            //----------------------------------------------------------------------------------
			pPerf->BeginEvent(L"Stochastic Depth Pass");

            //In Application, We Should Copy From Background Depth To Stochastic Depth
            //We Simplify This In The Sample.
            pd3dImmediateContext->ClearDepthStencilView(m_pStochasticDepth->pDSV, D3D11_CLEAR_DEPTH, 1.0, 0);

            pd3dImmediateContext->OMSetRenderTargets(0, NULL, m_pStochasticDepth->pDSV);
            pd3dImmediateContext->OMSetBlendState(m_pNoBlendBS, m_BlendFactor, 0XFFFFFFFF);
            pd3dImmediateContext->OMSetDepthStencilState(m_pDepthNoStencilDS, 0);

            pd3dImmediateContext->PSSetShader(m_pStochasticDepthPS, NULL, 0);
            pd3dImmediateContext->PSSetShaderResources(0, 1, &m_pRndTextureSRV);

            DrawMesh(pd3dImmediateContext, m_Mesh);

			pPerf->EndEvent();

            //----------------------------------------------------------------------------------
            // 3. We Merge TotalAlpha And Accumulate Together
            //----------------------------------------------------------------------------------
			pPerf->BeginEvent(L"TotalAlpha And Accumulate Pass");

            float ClearStochasticColorAndCorrectTotalAlpha[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            pd3dImmediateContext->ClearRenderTargetView(m_pStochasticColorAndCorrectTotalAlphaRenderTarget->pRTV, ClearStochasticColorAndCorrectTotalAlpha);

            float ClearStochasticTotalAlpha[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            pd3dImmediateContext->ClearRenderTargetView(m_pStochasticTotalAlphaRenderTarget->pRTV, ClearStochasticTotalAlpha);

			//UnBind Stochastic Depth
			//DSV->SRV
            ID3D11RenderTargetView *pRTVs[2] =
            {
            	m_pStochasticColorAndCorrectTotalAlphaRenderTarget->pRTV,
            	m_pStochasticTotalAlphaRenderTarget->pRTV
            };
            pd3dImmediateContext->OMSetRenderTargets(2, pRTVs, m_pBackgroundDepth->pDSV);

			pd3dImmediateContext->OMSetBlendState(m_pTotalAlphaAndAccumulateBS, m_BlendFactor, 0xffffffff);
            pd3dImmediateContext->OMSetDepthStencilState(m_pDepthNoWriteDS, 0);

			pd3dImmediateContext->PSSetShader(m_pTotalAlphaAndAccumulatePS, NULL, 0);

			ID3D11ShaderResourceView *pSRVs[1] =
			{
				m_pStochasticDepth->pSRV
			};
			pd3dImmediateContext->PSSetShaderResources(0, 1, pSRVs);


            DrawMesh(pd3dImmediateContext, m_Mesh);

			pPerf->EndEvent();
        }

        //----------------------------------------------------------------------------------
        // 5. Final full-screen pass, blending the transparent colors over the background
        //----------------------------------------------------------------------------------
		pPerf->BeginEvent(L"Composite Pass"); //Total Alpha Correction And Under Operator

        pd3dImmediateContext->OMSetRenderTargets(1, &pBackBuffer, NULL);
        pd3dImmediateContext->OMSetDepthStencilState(m_pNoDepthNoStencilDS, 0);
        pd3dImmediateContext->OMSetBlendState(m_pNoBlendBS, m_BlendFactor, 0xffffffff);

        pd3dImmediateContext->VSSetShader(m_pFullScreenTriangleVS, NULL, 0);
        pd3dImmediateContext->PSSetShader(m_pCompositePS, NULL, 0);

		ID3D11ShaderResourceView *pSRVs[3] =
		{
			m_pBackgroundRenderTarget->pSRV,
			m_pStochasticColorAndCorrectTotalAlphaRenderTarget->pSRV,
			m_pStochasticTotalAlphaRenderTarget->pSRV
		};
        pd3dImmediateContext->PSSetShaderResources(0, 3, pSRVs);

        pd3dImmediateContext->Draw(3, 0);

		//UnBind SRV->RTV
		ID3D11ShaderResourceView *pNULLSRVs[3] =
		{
			NULL,
			NULL,
			NULL
		};
		pd3dImmediateContext->PSSetShaderResources(0, 3, pNULLSRVs);


		pPerf->EndEvent();

		pPerf->Release();
    }

    void SetNumPasses(UINT NumPasses)
    {

    }

    ~StochasticTransparency()
    {
		SAFE_DELETE(m_pBackgroundRenderTarget);
		SAFE_DELETE(m_pBackgroundDepth);
		SAFE_DELETE(m_pStochasticDepth);
		SAFE_DELETE(m_pStochasticColorAndCorrectTotalAlphaRenderTarget);
        SAFE_DELETE(m_pStochasticTotalAlphaRenderTarget);
		SAFE_RELEASE(m_pStochasticDepthPS);
		SAFE_RELEASE(m_pTotalAlphaAndAccumulatePS);
		SAFE_RELEASE(m_pCompositePS);
        SAFE_RELEASE(m_pRndTexture);
        SAFE_RELEASE(m_pRndTextureSRV);
        SAFE_RELEASE(m_pTotalAlphaAndAccumulateBS);
        SAFE_RELEASE(m_pDepthNoWriteDS);

    }

protected:
    void CreateShaders(ID3D11Device* pd3dDevice)
    {
        HRESULT hr;

        V(pd3dDevice->CreatePixelShader(g_StochasticDepthPS, sizeof(g_StochasticDepthPS), NULL, &m_pStochasticDepthPS));

        V(pd3dDevice->CreatePixelShader(g_AccumulateAndTotalAlphaPS, sizeof(g_AccumulateAndTotalAlphaPS), NULL, &m_pTotalAlphaAndAccumulatePS));

        V(pd3dDevice->CreatePixelShader(g_CompositePS, sizeof(g_CompositePS), NULL, &m_pCompositePS));

    }

    void CreateBlendStates(ID3D11Device* pd3dDevice)
    {
        D3D11_BLEND_DESC BlendStateDesc;
        //To ensure uncorrelated, the AlphaToCoverage can't be used
        BlendStateDesc.AlphaToCoverageEnable = FALSE;

        //0 For Accumulate
        //1 For TotalAlpha
        BlendStateDesc.IndependentBlendEnable = TRUE;

        //DestColor/*ToRT*/ = BlendOp(DestBlend*DestColor/*FromRT*/, SrcBlend*SrcColor/*FromPS*/)
        //DestAlpha/*ToRT*/ = BlendOpAlpha(DestBlendAlpha*DestAlpha/*FromRT*/, SrcBlendAlpha*SrcAlpha/*FromPS*/)

        //StochasticColor+CorrectTotalAlpha
        BlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
        BlendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
        BlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        //StochasticTotalAlpha
        BlendStateDesc.RenderTarget[1].BlendEnable = TRUE;
        BlendStateDesc.RenderTarget[1].SrcBlend = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[1].DestBlend = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[1].BlendOp = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[1].SrcBlendAlpha = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[1].DestBlendAlpha = D3D11_BLEND_ONE;
        BlendStateDesc.RenderTarget[1].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[1].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        for (int i = 2; i < 8; ++i)
        {
        	BlendStateDesc.RenderTarget[i].BlendEnable = FALSE;
        	BlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        }

        pd3dDevice->CreateBlendState(&BlendStateDesc, &m_pTotalAlphaAndAccumulateBS);
    }

    void CreateRandomBitmasks(ID3D11Device* pd3dDevice)
    {
        MTRand rng;
        rng.seed((unsigned)0);

        int numbers[NUM_MSAA_SAMPLES];
        unsigned int *allmasks = new unsigned int[RANDOM_SIZE * (ALPHA_VALUES + 1)];

        for (int y = 0; y <= ALPHA_VALUES; y++) // Inclusive, we need alpha = 1.0
        {
            for (int x = 0; x < RANDOM_SIZE; x++)
            {
                // Initialize array
                for (int i = 0; i < NUM_MSAA_SAMPLES; i++)
                {
                    numbers[i] = i;
                }

                // Scramble!
                for (int i = 0; i < NUM_MSAA_SAMPLES * 2; i++)
                {
                    std::swap(numbers[rng.randInt() % NUM_MSAA_SAMPLES], numbers[rng.randInt() % NUM_MSAA_SAMPLES]);
                }

                // Create the mask
                unsigned int mask = 0;
                float nof_bits_to_set = (float(y) / float(ALPHA_VALUES)) * NUM_MSAA_SAMPLES;
                for (int bit = 0; bit < int(nof_bits_to_set); bit++)
                {
                    mask |= (1 << numbers[bit]);
                }
                float prob_of_last_bit = (nof_bits_to_set - floor(nof_bits_to_set));
                if (rng.randExc() < prob_of_last_bit)
                {
                    mask |= (1 << numbers[int(nof_bits_to_set)]);
                }

                allmasks[y * RANDOM_SIZE + x] = mask;
            }
        }

        D3D11_TEXTURE2D_DESC texDesc;
        texDesc.Width            = RANDOM_SIZE;
        texDesc.Height           = ALPHA_VALUES + 1;
        texDesc.MipLevels        = 1;
        texDesc.ArraySize        = 1;
        texDesc.Format           = DXGI_FORMAT_R32_UINT;
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Usage            = D3D11_USAGE_IMMUTABLE;
        texDesc.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
        texDesc.CPUAccessFlags   = 0;
        texDesc.MiscFlags        = 0;

        D3D11_SUBRESOURCE_DATA srDesc;
        srDesc.pSysMem          = allmasks;
        srDesc.SysMemPitch      = texDesc.Width * sizeof(unsigned int);
        srDesc.SysMemSlicePitch = 0;

        SAFE_RELEASE(m_pRndTexture);
        pd3dDevice->CreateTexture2D(&texDesc, &srDesc, &m_pRndTexture);

        SAFE_RELEASE(m_pRndTextureSRV);
        pd3dDevice->CreateShaderResourceView(m_pRndTexture, NULL, &m_pRndTextureSRV);

        delete[] allmasks;
    }

    void CreateFrameBuffer(ID3D11Device* pd3dDevice, UINT Width, UINT Height)
    {
		//Turning on MSAA in StochasticDepthPass is to random sample
		//and the stochastic transparency intrinsically doesn't demand other passes to turn on the MSAA.
        {
        D3D11_TEXTURE2D_DESC texDesc;
        texDesc.Width = Width;
        texDesc.Height = Height;
        texDesc.ArraySize = 1;
        texDesc.MiscFlags = 0;
        texDesc.MipLevels = 1;
        texDesc.SampleDesc.Count = 1U;
        texDesc.SampleDesc.Quality = 0U;
        texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.CPUAccessFlags = NULL;

        m_pBackgroundRenderTarget = new SimpleRT(pd3dDevice, &texDesc, DXGI_FORMAT_R8G8B8A8_UNORM);
        m_pStochasticColorAndCorrectTotalAlphaRenderTarget = new SimpleRT(pd3dDevice, &texDesc, DXGI_FORMAT_R8G8B8A8_UNORM);
        m_pStochasticTotalAlphaRenderTarget = new SimpleRT(pd3dDevice, &texDesc, DXGI_FORMAT_R16_FLOAT); //STOCHASTIC_COLOR_FORMAT;
        }

        {
        D3D11_TEXTURE2D_DESC texDesc;
        texDesc.ArraySize = 1;
        texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        texDesc.CPUAccessFlags = NULL;
        texDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        texDesc.Width = Width;
        texDesc.Height = Height;
        texDesc.MipLevels = 1;
        texDesc.MiscFlags = NULL;
        texDesc.SampleDesc.Count = 1U;
        texDesc.SampleDesc.Quality = 0U;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        m_pBackgroundDepth = new SimpleDepthStencil(pd3dDevice, &texDesc);
        }
    }

    void CreateStochasticDepth(ID3D11Device* pd3dDevice, UINT Width, UINT Height)
    {
        m_pStochasticDepth = new StochasticDepth(pd3dDevice, Width, Height);
    }
	
	SimpleRT *m_pBackgroundRenderTarget;
	SimpleDepthStencil *m_pBackgroundDepth;
    StochasticDepth* m_pStochasticDepth;
	SimpleRT *m_pStochasticColorAndCorrectTotalAlphaRenderTarget;
    SimpleRT *m_pStochasticTotalAlphaRenderTarget;

	ID3D11PixelShader *m_pStochasticDepthPS;
	ID3D11PixelShader *m_pTotalAlphaAndAccumulatePS;
	ID3D11PixelShader *m_pCompositePS;

	ID3D11Texture2D *m_pRndTexture;
	ID3D11ShaderResourceView *m_pRndTextureSRV;

	ID3D11BlendState *m_pTotalAlphaAndAccumulateBS;
};
