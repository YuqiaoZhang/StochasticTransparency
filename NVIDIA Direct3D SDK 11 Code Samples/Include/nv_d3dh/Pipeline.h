#pragma once

#include <dxgi.h>
#include <d3d11.h>
#include <d3dCompiler.h>
#include <d3dx11.h>
#include <d3dx11effect.h>
#include <dxerr.h>

#include <map>
#include <stack>

#include "Primitives.h"
#include "Buffer.h"

namespace d3d11
{
    // use this macro definitions to safe calls
    //
    #define BindSRV(str,tex) BindSRV_F((str),(tex),__FILE__,__LINE__)
    #define BindUAV(str,tex)  BindUAV_F((str),(tex),__FILE__,__LINE__)
	  #define BindUAVAppend(str,tex)  BindUAVAppend_F((str),(tex),__FILE__,__LINE__)
    #define ApplyTechnique(name, pass) ApplyTechnique_F((name),(pass),__FILE__,__LINE__)
    #define SetShaderVariable(name, val) SetShaderVariable_F((name), (val), __FILE__, __LINE__)
    #define SetShaderVariableScalarArray(name, val, size) SetShaderVariableScalarArray_F((name), (val), (size), __FILE__, __LINE__)
    #define SetShaderVariableVectorArray(name, val, size) SetShaderVariableVectorArray_F((name), (val), (size), __FILE__, __LINE__)

    // a very simple class that represent d3d pipeline
    //
    class Pipeline
    {
    public:
        
        Pipeline(ID3D11Device* a_pDevice, ID3DX11Effect* a_pEffect);
        Pipeline(ID3D11Device* a_pDevice, ID3DX11Effect* a_pEffect, FullScreenQuad* a_quad);
        Pipeline(ID3D11Device* a_pDevice, const wchar_t* a_fxFileName);
        virtual ~Pipeline();

        // 
        //
        void BindSRV_F(std::string a_name, ID3D11Texture2D* a_pTex, const char* file, int line);
        void BindSRV_F(std::string a_name, ID3D11Texture3D* a_pTex, const char* file, int line);
        void BindSRV_F(std::string a_name, ID3D11Buffer* a_pBuff, const char* file, int line);
        void BindSRV_F(std::string a_name, IBuffer* a_pBuff, const char* file, int line);

        void BindUAV_F(std::string a_name, ID3D11Texture3D* a_pTex, const char* file, int line);
        void BindUAV_F(std::string a_name, ID3D11Texture2D* a_pTex, const char* file, int line);
        void BindUAV_F(std::string a_name, ID3D11Buffer* a_pBuff, const char* file, int line);
				void BindUAVAppend_F(std::string a_name, ID3D11Buffer* a_pBuff, const char* file, int line);
        void BindUAV_F(std::string a_name, IBuffer* a_pBuff, const char* file, int line);

        void UnbindAllUAV();
        void UnbindAllSRV();

				void Dispatch(int x,int y, int z) { m_pContext->Dispatch(x,y,z); }
				void Draw(int a_vertexCount, int a_startVertexLocation) { m_pContext->Draw(a_vertexCount, a_startVertexLocation);}

        //
        //
        void SetShaderVariable_F(std::string a_name, float val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, int val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const int* val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const float* val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const D3DXVECTOR2& val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const D3DXVECTOR3& val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const D3DXVECTOR4& val, const char* file, int line);
        void SetShaderVariable_F(std::string a_name, const D3DXMATRIX& val, const char* file, int line);
        
        void SetShaderVariableScalarArray_F(std::string a_name, const float* arr, int size, const char* file, int line);
        void SetShaderVariableVectorArray_F(std::string a_name, const float* arr, int size, const char* file, int line);

        void ApplyTechnique_F(const char* name, int passm, const char* file, int line);

        void BeginRenderingToTexture(ID3D11Texture2D* a_pTex, int X=0, int Y=0, int Z=0);
        void EndRenderingToTexture();

        void BeginRenderingTo3DTexture(ID3D11Texture3D* a_pTex, int X=0, int Y=0, int Z=0);
        void EndRenderingTo3DTexture();

        enum DS_STATES{DEFAULT = 1, DISABLE_DEPTH_TEST = 2};
        void SetDepthStencilState(int flag);

        // delete appropriate view from the appropriate cache
        //
        void ReleaseRTV(ID3D11Resource* res);
        void ReleaseSRV(ID3D11Resource* res);
        void ReleaseUAV(ID3D11Resource* res);
        void ReleaseAllViews(ID3D11Resource* res);

				ID3D11ShaderResourceView*  GetSRV(ID3D11Resource* key);
				ID3D11RenderTargetView*    GetRTV(ID3D11Resource* key);
				ID3D11UnorderedAccessView* GetUAV(ID3D11Resource* key);

        // trash RTV, SRV and AUV caches
        //
        void FreeCaches();

        void RunOldSchoolKernel(const char* name, ID3D11Texture2D* output, int X=0, int Y=0, int Z=0);
        void RunKernel(const char* name, int X, int Y, int Z);

        void CopyBufferToTexture2D(ID3D11Texture2D* a_pTex, ID3D11Buffer* a_pBuff);
        void CopyTexture2DToBuffer(ID3D11Buffer* a_pBuff, ID3D11Texture2D* a_pTex);

        FullScreenQuad* GetFullScreenQuad() {return m_pFullScreenQuad;}

        ID3D11Device*        GetDevice() {return m_pDevice;}
        ID3D11DeviceContext* GetContext() {return m_pContext;}
        ID3DX11Effect*       GetEffect() {return m_pEffect;}

        ID3D11UnorderedAccessView* GetCachedUAV_View(ID3D11Texture2D* a_pTex)
            { return m_UAVCache[a_pTex]; }

        // WRNING! usin this functions is unsafe
        //
        //void PushEffect(ID3DX11Effect* a_effect);
        //ID3DX11Effect* PopEffect();

    protected:
        ID3D11Device*           m_pDevice;
        ID3D11DeviceContext*    m_pContext;
        ID3DX11Effect*          m_pEffect;
        ID3D11Query*            m_pQuery;

        FullScreenQuad*         m_pFullScreenQuad;

        float m_clearColor[4];

        // a temp variables for rendering to texture
        //
        D3D11_TEXTURE2D_DESC m_oldTexDesc;
        D3D11_VIEWPORT m_OldVP;
        ID3D11RenderTargetView* m_apOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
        ID3D11DepthStencilView* m_pOldDS;

        ID3D11RenderTargetView* m_pTexRTV;
        ID3D11DepthStencilView* m_pTexDSV;
        
        ID3D11RenderTargetView* m_pTex3DRTV;
        ID3D11Texture2D* m_pDepthTex;

        // try to cache RTV's, SRV's, UAV's.
        //
        typedef std::map<ID3D11Resource*,ID3D11RenderTargetView*> RTVCache;
        typedef std::map<ID3D11Resource*,ID3D11ShaderResourceView*> SRVCache;
        typedef std::map<ID3D11Resource*,ID3D11UnorderedAccessView*> UAVCache;

        RTVCache m_RTVCache;
        SRVCache m_SRVCache;
        UAVCache m_UAVCache;

        virtual void CreateRenderToTextureResources(ID3D11Texture2D* a_pTex,const char* file, int line);
        virtual void CreateRenderToTextureResources(ID3D11Texture3D* a_pTex);
        bool SameTextureDescription(const D3D11_TEXTURE2D_DESC a, const D3D11_TEXTURE2D_DESC& b);
   
    };


}

