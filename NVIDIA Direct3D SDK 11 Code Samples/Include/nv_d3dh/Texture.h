#pragma once

#include "Buffer.h"

namespace d3d11
{


    template<class T>
    static ID3D11Texture2D* CreateTexture2D(ID3D11Device* pDevice, int width, int height, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0)
    {
        ID3D11Texture2D* pTexture = 0;

        DXGI_SAMPLE_DESC simpleSample;
        simpleSample.Count = 1;
        simpleSample.Quality = 0;

        D3D11_TEXTURE2D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE2D_DESC));
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGIFormat<T,32>();
        texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;	
				texDesc.MiscFlags = a_miscFlags;

        DX_SAFE_CALL(pDevice->CreateTexture2D(&texDesc, 0, &pTexture));

        return pTexture;
    }

    template<class T>
    static ID3D11Texture2D* CreateTexture2D(ID3D11Device* pDevice, void* a_initData, int width, int height,int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0)
    {
        ID3D11Texture2D* pTexture = 0;

        DXGI_SAMPLE_DESC simpleSample;
        simpleSample.Count = 1;
        simpleSample.Quality = 0;

        D3D11_TEXTURE2D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE2D_DESC));
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGIFormat<T,32>();
        texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = a_miscFlags;

        D3D11_SUBRESOURCE_DATA initData;
        ZeroMemory(&initData, sizeof(D3D11_SUBRESOURCE_DATA));
        initData.pSysMem = a_initData;
        initData.SysMemPitch = width*sizeof(T);

        DX_SAFE_CALL(pDevice->CreateTexture2D(&texDesc, &initData, &pTexture));
        return pTexture;
    }

    template<class T>
    static ID3D11Texture2D* CreateTexture2D(ID3D11Device* pDevice, const String& fileName)
    {
        ID3D11Texture2D* pTexture = 0;

        ID3D11Resource* pTextRes = 0;
        DX_SAFE_CALL(D3DX11CreateTextureFromFile(pDevice, fileName.c_str(), NULL, NULL, &pTextRes, NULL ));
        DX_SAFE_CALL(pTextRes->QueryInterface(__uuidof(ID3D11Texture2D), (LPVOID*) &pTexture));
        SAFE_RELEASE(pTextRes);

        return pTexture;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////
    //// untested
    template<class T>
    static ID3D11Texture1D* CreateTexture1D(ID3D11Device* pDevice, void* a_initData, int size,int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0)
    {
        ID3D11Texture1D* pTexture = 0;

        D3D11_TEXTURE1D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE1D_DESC));
        texDesc.Width = size;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGIFormat<T,32>();
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = a_miscFlags;

        D3D11_SUBRESOURCE_DATA initData;
        ZeroMemory(&initData, sizeof(D3D11_SUBRESOURCE_DATA));
        initData.pSysMem = a_initData;
        initData.SysMemPitch = 0; //width*sizeof(T);

        DX_SAFE_CALL(pDevice->CreateTexture1D(&texDesc, &initData, &pTexture));
        return pTexture;
    }

    template<class T>
    static ID3D11Texture1D* CreateTexture1D(ID3D11Device* pDevice, int size, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0)
    {
        ID3D11Texture1D* pTexture = 0;

        D3D11_TEXTURE1D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE1D_DESC));
        texDesc.Width = size;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGIFormat<T,32>();
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = a_miscFlags;

        DX_SAFE_CALL(pDevice->CreateTexture1D(&texDesc, 0, &pTexture));
        return pTexture;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    ////
    template<class T, int bits>
    static ID3D11Texture3D* CreateTexture3D(ID3D11Device* pDevice, int width, int height, int depth, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0)
    {
        ID3D11Texture3D* pTexture = 0;

        //DXGI_SAMPLE_DESC simpleSample;
        //simpleSample.Count = 1;
        //simpleSample.Quality = 0;

        D3D11_TEXTURE3D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));

        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.Depth = depth;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGIFormat<T,bits>();
        //texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = a_miscFlags;

        DX_SAFE_CALL(pDevice->CreateTexture3D(&texDesc, 0, &pTexture));

        return pTexture;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    ////
    template<class T, int bits>
    static ID3D11Texture3D* CreateTexture3D(ID3D11Device* pDevice, void* a_initData, 
                                            int width, int height, int depth, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE)
    {
        ID3D11Texture3D* pTexture = 0;


        D3D11_TEXTURE3D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));

        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.Depth = depth;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGIFormat<T,bits>();
        //texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA initData;
        ZeroMemory(&initData, sizeof(D3D11_SUBRESOURCE_DATA));
        initData.pSysMem = a_initData;
        initData.SysMemPitch = width*sizeof(T);
        initData.SysMemSlicePitch = width*height*sizeof(T);

        DX_SAFE_CALL(pDevice->CreateTexture3D(&texDesc, &initData, &pTexture));

        return pTexture;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////
    ////
    /*template<class T>
    static ID3D11Texture3D* CreateTexture3D_16Bit(ID3D11Device* pDevice, 
                                                  int width, int height, int depth, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE)
    {
        ID3D11Texture3D* pTexture = 0;

        D3D11_TEXTURE3D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));

        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.Depth = depth;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGIFormat<T,16>();
        //texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;
        texDesc.MiscFlags = 0;

        DX_SAFE_CALL(pDevice->CreateTexture3D(&texDesc, NULL, &pTexture));

        return pTexture;
    }*/

		template<class T>
    static ID3D11Texture2D* CreateTexture2D_WithMipLevels(ID3D11Device* pDevice, int width, int height, int a_bindFlags = D3D11_BIND_SHADER_RESOURCE, int a_miscFlags = 0, int a_mipLevels = 10)
    {
        ID3D11Texture2D* pTexture = 0;

        DXGI_SAMPLE_DESC simpleSample;
        simpleSample.Count = 1;
        simpleSample.Quality = 0;

        D3D11_TEXTURE2D_DESC texDesc;
        ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE2D_DESC));
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.MipLevels = a_mipLevels;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGIFormat<T,32>();
        texDesc.SampleDesc = simpleSample;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = a_bindFlags;
        texDesc.CPUAccessFlags = 0;	
				texDesc.MiscFlags = a_miscFlags;

        DX_SAFE_CALL(pDevice->CreateTexture2D(&texDesc, 0, &pTexture));

        return pTexture;
    }

}


