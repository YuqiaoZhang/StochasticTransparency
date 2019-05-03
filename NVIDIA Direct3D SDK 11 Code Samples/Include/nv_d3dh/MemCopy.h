#pragma once 
#include <dxgi.h>
#include <d3d11.h>
#include <d3dCompiler.h>
#include <d3dx11.h>
#include <d3dx11effect.h>
#include <dxerr.h>
#include "D3DHelpers.h"

namespace d3d11
{

    template<class T> char*		_HOST(T* src) {return (char*)src;}
    template<class T> const char* _HOST(const T* src) {return (const char*)src;}

    #define ExtendedMemCopy(dst,src,size,device) ExtendedMemCopyF(dst, src, size, device, __FILE__, __LINE__)  

    template<class TO, class FROM>
    static void ExtendedMemCopyF(TO* dst, const FROM* src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {ERROR_UNDEFINED_TYPES_FOR_MEMCOPY;}

    template<>
    static void ExtendedMemCopyF<ID3D11Texture2D, char>(ID3D11Texture2D* dst, const char* src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11Texture2D* tempTexture = 0;
        ID3D11DeviceContext* pd3dImmediateContext = GetCtx(pDevice);  

        try
        {
            if(src == 0)
                throw std::runtime_error("zero input (source) pointer");

            if(dst == 0)
                throw std::runtime_error("zero output (destination) texture");

            D3D11_TEXTURE2D_DESC texDesc;
            dst->GetDesc(&texDesc);

            D3D11_SUBRESOURCE_DATA initData;
            ZeroMemory(&initData, sizeof(D3D11_SUBRESOURCE_DATA));
            initData.pSysMem = src;
            initData.SysMemPitch = texDesc.Width*SizeOfDeviceFormat(texDesc.Format);

            if(size!=texDesc.Width*texDesc.Height*SizeOfDeviceFormat(texDesc.Format))
                throw std::runtime_error("incorrect input size ");

            DX_SAFE_CALL(pDevice->CreateTexture2D(&texDesc, &initData, &tempTexture));
            pd3dImmediateContext->CopyResource(dst,tempTexture);

            SAFE_RELEASE(tempTexture);
            SAFE_RELEASE(pd3dImmediateContext);
        }
        catch(std::runtime_error e)
        {
           SAFE_RELEASE(tempTexture);
           SAFE_RELEASE(pd3dImmediateContext);
           RUN_TIME_ERROR_AT((std::string("ExtendedMemCopy: 'Host memory' -> 'Texture2D' : failed bacause ") + std::string(e.what())).c_str(), file, line);
        }
        catch(...)
        {
           SAFE_RELEASE(tempTexture);
           SAFE_RELEASE(pd3dImmediateContext);
           RUN_TIME_ERROR_AT("Unexpected exeption in ExtendedMemCopy", file, line);
        }

    }


    template<>
    static void ExtendedMemCopyF<char, ID3D11Texture2D>(char* dst, const ID3D11Texture2D* a_src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11Texture2D* tempTexture = 0;
        ID3D11DeviceContext* pd3dImmediateContext = GetCtx(pDevice);  

        try
        {
            ID3D11Texture2D* src = const_cast<ID3D11Texture2D*>(a_src);

            if(src == 0)
                throw std::runtime_error("zero input (source) texture");

            if(dst == 0)
                throw std::runtime_error("zero output (destination) pointer");

            ID3D11Texture2D* tempTexture;
            D3D11_TEXTURE2D_DESC texDesc;
            D3D11_MAPPED_SUBRESOURCE MappedResource;

            src->GetDesc(&texDesc);

            texDesc.Usage = D3D11_USAGE_STAGING;
            texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            texDesc.BindFlags = 0;

            if(size!=texDesc.Width*texDesc.Height*SizeOfDeviceFormat(texDesc.Format))
                throw std::runtime_error("incorrect input size");

            DX_SAFE_CALL(pDevice->CreateTexture2D(&texDesc, 0, &tempTexture));

            pd3dImmediateContext->CopyResource(tempTexture, src);

            DX_SAFE_CALL(pd3dImmediateContext->Map(tempTexture, 0, D3D11_MAP_READ, 0, &MappedResource ));

            memcpy(dst, MappedResource.pData, size*sizeof(char) );

            pd3dImmediateContext->Unmap(tempTexture, 0 );

            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
        }
        catch(std::runtime_error e)
        {
            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
            RUN_TIME_ERROR_AT((std::string("ExtendedMemCopy: 'Texture2D' -> 'Host memory' : failed bacause ") + std::string(e.what())).c_str(), file, line);
        }
        catch(...)
        {
            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
            RUN_TIME_ERROR_AT("Unexpected exeption in ExtendedMemCopy", file, line);
        }

    }

    template<>
    static void ExtendedMemCopyF<ID3D11Texture2D, ID3D11Texture2D>(ID3D11Texture2D* dst, const ID3D11Texture2D* a_src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11DeviceContext* pContext = GetCtx(pDevice);  

        try
        {
            ID3D11Texture2D* src = const_cast<ID3D11Texture2D*>(a_src);

            D3D11_TEXTURE2D_DESC texDescSrc,texDescDst;

            dst->GetDesc(&texDescSrc);
            src->GetDesc(&texDescDst);

            if(texDescSrc.Width != texDescDst.Width || texDescSrc.Height != texDescDst.Height)
                RUN_TIME_ERROR_AT("ExtendedMemCopy: Texture2D -> Texture2D : the size (w,h) of input and output texture are not match", file, line);

            //if(texDescSrc.Width*texDescSrc.Height*SizeOfDeviceFormat(texDescSrc.Format) != size)
              //  RUN_TIME_ERROR_AT("ExtendedMemCopy: Texture2D -> Texture2D : w*h!=size", file, line);

            pContext->CopyResource(dst, src);

            SAFE_RELEASE(pContext);
        }
        catch(...)
        {
            SAFE_RELEASE(pContext);
            throw;
        }
    }


    template<>
    static void ExtendedMemCopyF<ID3D11Buffer, ID3D11Buffer>(ID3D11Buffer* dst, const ID3D11Buffer* a_src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11DeviceContext* pContext = GetCtx(pDevice);  

        try
        {
            ID3D11Buffer* src = const_cast<ID3D11Buffer*>(a_src);

            D3D11_BUFFER_DESC descSrc,descDst;

            dst->GetDesc(&descSrc);
            src->GetDesc(&descDst);

            if(descSrc.ByteWidth != descDst.ByteWidth)
                RUN_TIME_ERROR_AT("ExtendedMemCopy: Buffer -> Buffer : the size of input and output buffer are not match", file, line);

            //if(descSrc.ByteWidth != size)
              //  RUN_TIME_ERROR_AT("ExtendedMemCopy: Buffer -> Buffer : buffer size not equal input size", file, line);

            pContext->CopyResource(dst, src);

            SAFE_RELEASE(pContext);
        }
        catch(...)
        {
            SAFE_RELEASE(pContext);
            throw;
        }
    }


		template<>
    static void ExtendedMemCopyF<char, ID3D11Buffer>(char* dst, const ID3D11Buffer* a_src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11DeviceContext* pContext = GetCtx(pDevice);  

        try
        {
            ID3D11Buffer* src = const_cast<ID3D11Buffer*>(a_src);
						ID3D11Buffer* tempBuffer;

						D3D11_BUFFER_DESC bufferDesc;
						ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
						bufferDesc.Usage            = D3D11_USAGE_STAGING;
						bufferDesc.ByteWidth        = size;
						bufferDesc.BindFlags        = 0;
						bufferDesc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

						DX_SAFE_CALL(pDevice->CreateBuffer(&bufferDesc, 0, &tempBuffer));
						pContext->CopyResource(tempBuffer, src);

						D3D11_MAPPED_SUBRESOURCE MappedResource;
						DX_SAFE_CALL(pContext->Map(tempBuffer, 0, D3D11_MAP_READ, 0, &MappedResource ));

						memcpy(dst, MappedResource.pData, size);

						pContext->Unmap(tempBuffer, 0 );

						SAFE_RELEASE(tempBuffer);
						
            SAFE_RELEASE(pContext);
        }
        catch(...)
        {
            SAFE_RELEASE(pContext);
            throw;
        }
    }

		template<>
    static void ExtendedMemCopyF<ID3D11Buffer, char>(ID3D11Buffer* a_src, const char* a_cpuData, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11DeviceContext* pContext = GetCtx(pDevice);  

        try
        {
						D3D11_BUFFER_DESC thisBufferDesc;
						a_src->GetDesc(&thisBufferDesc);

						if (thisBufferDesc.Usage == D3D11_USAGE_DEFAULT)
						{
							pContext->UpdateSubresource(a_src,0,0,a_cpuData,0,0);
						}
						else if (thisBufferDesc.Usage == D3D11_USAGE_DYNAMIC)
						{
							RUN_TIME_ERROR_AT("untested copy for dynamic buffers", file, line);
							D3D11_MAPPED_SUBRESOURCE res;
        
							pContext->Map(a_src,0,D3D11_MAP_WRITE,0,&res);
							
							int size2 = size/4;
							ASSERT(size%4==0);

							int* arr = (int*)res.pData;
							for(int i=0;i<size2/4;i+=4)
							{
								arr[i+0] = a_cpuData[i+0];
								arr[i+1] = a_cpuData[i+1];
								arr[i+2] = a_cpuData[i+2];
								arr[i+3] = a_cpuData[i+3];
							}

							for(int i= (size2/4)*4; i<size2; i++)
								arr[i] = a_cpuData[i];

							pContext->Unmap(a_src, 0);    
						}
						else
						{
							ID3D11Buffer* tempBuffer;
							D3D11_BUFFER_DESC bufferDesc;
							ZeroMemory( &bufferDesc, sizeof(bufferDesc) );

							bufferDesc.Usage     = D3D11_USAGE_DEFAULT;
							bufferDesc.ByteWidth = size;

							D3D11_SUBRESOURCE_DATA InitData;
							InitData.pSysMem = a_cpuData;
							InitData.SysMemPitch = 0;
							InitData.SysMemSlicePitch = 0;

							DX_SAFE_CALL(pDevice->CreateBuffer(&bufferDesc, &InitData, &tempBuffer));

							pContext->CopyResource(a_src, tempBuffer);

							SAFE_RELEASE(tempBuffer);
						}
					
            SAFE_RELEASE(pContext);
        }
        catch(...)
        {
            SAFE_RELEASE(pContext);
            throw;
        }
    }


    template<>
    static void ExtendedMemCopyF<char, ID3D11Texture3D>(char* dst, const ID3D11Texture3D* a_src, unsigned int size, ID3D11Device* pDevice, const char* file, int line)
    {
        ID3D11Texture3D* tempTexture = 0;
        ID3D11DeviceContext* pd3dImmediateContext = GetCtx(pDevice);  

        try
        {
            ID3D11Texture3D* src = const_cast<ID3D11Texture3D*>(a_src);

            if(src == 0)
                throw std::runtime_error("zero input (source) texture");

            if(dst == 0)
                throw std::runtime_error("zero output (destination) pointer");

            ID3D11Texture3D* tempTexture;
            D3D11_TEXTURE3D_DESC texDesc;
            D3D11_MAPPED_SUBRESOURCE MappedResource;

            src->GetDesc(&texDesc);

            texDesc.Usage = D3D11_USAGE_STAGING;
            texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            texDesc.BindFlags = 0;

            if(size!=texDesc.Width*texDesc.Height*texDesc.Depth*SizeOfDeviceFormat(texDesc.Format))
                throw std::runtime_error("incorrect input size");

            DX_SAFE_CALL(pDevice->CreateTexture3D(&texDesc, 0, &tempTexture));

            pd3dImmediateContext->CopyResource(tempTexture, src);

            DX_SAFE_CALL(pd3dImmediateContext->Map(tempTexture, 0, D3D11_MAP_READ, 0, &MappedResource ));

            memcpy(dst, MappedResource.pData, size);

            pd3dImmediateContext->Unmap(tempTexture, 0 );

            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
        }
        catch(std::runtime_error e)
        {
            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
            RUN_TIME_ERROR_AT((std::string("ExtendedMemCopy: 'Texture2D' -> 'Host memory' : failed bacause ") + std::string(e.what())).c_str(), file, line);
        }
        catch(...)
        {
            SAFE_RELEASE(pd3dImmediateContext);
            SAFE_RELEASE(tempTexture);
            RUN_TIME_ERROR_AT("Unexpected exeption in ExtendedMemCopy", file, line);
        }

    }



}