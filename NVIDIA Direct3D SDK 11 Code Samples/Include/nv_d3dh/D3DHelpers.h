//============================================================================
//  
//============================================================================
#pragma once
#pragma warning (disable:4996)

#include <dxgi.h>
#include <d3d11.h>
#include <d3dCompiler.h>
#include <d3dx11.h>
#include <d3dx11effect.h>
#include <dxerr.h>
#include <d3dx10math.h>

#include <assert.h>
#include <iostream>
#include <string>
#include <wchar.h>
#include <tchar.h>
#include <windows.h>

std::wstring ToWideString(const char* pStr);
std::string ToNarrowString(const wchar_t* pStr);

void ThrowStdRuntimeError(const char* file, int line, const char* err_msg);
void ThrowStdRuntimeErrorU(const wchar_t* file, int line, const wchar_t* err_msg);

//#ifndef UNICODE

#define RUN_TIME_ERROR(str) ThrowStdRuntimeError(__FILE__, __LINE__, (str)) 
#define RUN_TIME_ERROR_AT(str, file, line) ThrowStdRuntimeError(file,line, (str))

#ifndef NDEBUG
#define ASSERT(_Expression) if(!(_Expression)) { RUN_TIME_ERROR("Assertion failed"); }
#else
#define ASSERT(_Expression)
#endif

/*#else

#define RUN_TIME_ERROR(str) ThrowStdRuntimeErrorU(TEXT(__FILE__), __LINE__, str) 
#define RUN_TIME_ERROR_AT(str, file, line) ThrowStdRuntimeErrorU(file,(line), (str))

#ifndef NDEBUG
#define ASSERT(_Expression) if(!(_Expression)) { RUN_TIME_ERROR("Assertion failed"); }
#else
#define ASSERT(_Expression)
#endif

#endif*/


#define DX_SAFE_CALL(func_call) if(FAILED((func_call))) RUN_TIME_ERROR("DX call failed")
// make our own later
//#define CASSERT(_Expression, Message) (void)( (!!(_Expression)) || (_wassert(_CRT_WIDE(#_Expression), _CRT_WIDE(__FILE__), __LINE__), 0) )


typedef unsigned int uint;
typedef unsigned char uchar;


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif


namespace d3d11
{

    template<bool>
    struct STATIC_ASSERT;

    template<>
    struct STATIC_ASSERT<true> {};

    template<bool condition,class Then,class Else>
    struct IF
    {typedef Then RET;
    };

    template<class Then,class Else>
    struct IF<false,Then,Else>
    { typedef Else RET;
    };



#ifndef UNICODE
    typedef std::string String;
#else
    typedef std::wstring String;
#endif


    static ID3D11DeviceContext* GetCtx(ID3D11Device* pDev)	
    {
        if(pDev==0)
            RUN_TIME_ERROR("GetCtx: zero device pointer");

        ID3D11DeviceContext* pd3dImmediateContext = 0;  
        pDev->GetImmediateContext(&pd3dImmediateContext);

        if(pd3dImmediateContext==0)
            RUN_TIME_ERROR("GetImmediateContext failed");

        return pd3dImmediateContext;
    }


    template<class T, int bits>
    static DXGI_FORMAT DXGIFormat() { return DXGI_FORMAT_R32_FLOAT; }

    template<>
    static DXGI_FORMAT DXGIFormat<float,32>()       {return DXGI_FORMAT_R32_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<D3DXVECTOR2,32>() {return DXGI_FORMAT_R32G32_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<D3DXVECTOR3,32>() {return DXGI_FORMAT_R32G32B32_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<D3DXVECTOR4,32>() {return DXGI_FORMAT_R32G32B32A32_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<int,32>()         {return DXGI_FORMAT_R32_SINT;}

    template<>
    static DXGI_FORMAT DXGIFormat<unsigned int,32>() {return DXGI_FORMAT_R32_UINT;}

    template<>
    static DXGI_FORMAT DXGIFormat<char,32>()         {return DXGI_FORMAT_R8_SINT;}

    template<>
    static DXGI_FORMAT DXGIFormat<unsigned char,32>() {return DXGI_FORMAT_R8_UINT;}

		template<>
    static DXGI_FORMAT DXGIFormat<float,16>()       {return DXGI_FORMAT_R16_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<D3DXVECTOR2,16>() {return DXGI_FORMAT_R16G16_FLOAT;}

    //template<>
    //static DXGI_FORMAT DXGIFormat<D3DXVECTOR3>() {return DXGI_FORMAT_R16G16B16_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<D3DXVECTOR4,16>() {return DXGI_FORMAT_R16G16B16A16_FLOAT;}

    template<>
    static DXGI_FORMAT DXGIFormat<int,16>()         {return DXGI_FORMAT_R16_SINT;}

    template<>
    static DXGI_FORMAT DXGIFormat<unsigned int,16>() {return DXGI_FORMAT_R16_UINT;}

    static int SizeOfDeviceFormat(DXGI_FORMAT format)
    {
        switch(format)
        {
        case DXGI_FORMAT_R32_FLOAT: return sizeof(float);
        case DXGI_FORMAT_R32G32_FLOAT: return sizeof(float)*2;
        case DXGI_FORMAT_R32G32B32_FLOAT: return sizeof(float)*3;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return sizeof(float)*4;
        case DXGI_FORMAT_R32_SINT: return sizeof(int);
        case DXGI_FORMAT_R32_UINT: return sizeof(unsigned int);

        case DXGI_FORMAT_R16_FLOAT: return sizeof(float)/2;
        case DXGI_FORMAT_R16G16_FLOAT: return sizeof(float);
        //case DXGI_FORMAT_R16G16B16_FLOAT: return sizeof(float)*3/2;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return sizeof(float)*2;
        case DXGI_FORMAT_R16_SINT: return sizeof(short);
        case DXGI_FORMAT_R16_UINT: return sizeof(short);

        case DXGI_FORMAT_R8_SINT: return sizeof(char);
        case DXGI_FORMAT_R8_UINT: return sizeof(unsigned char);
        default: RUN_TIME_ERROR("Unknown Device Format"); return 0;
        };
    }

    //template<class T>
    //static DXGI_FORMAT DXGIFormat16() { return DXGI_FORMAT_R32_FLOAT; }


    static D3DXMATRIX ComputeNormalMatrix(const D3DXMATRIX& aWorld)
    {
        D3DXMATRIX matrixNorm = aWorld, tmp;

        matrixNorm._41=0;
        matrixNorm._42=0;
        matrixNorm._43=0;
        matrixNorm._44=1;

        D3DXMatrixTranspose(&tmp,&matrixNorm);
        D3DXMatrixInverse(&matrixNorm,0,&tmp);

        return matrixNorm;
    }

}

