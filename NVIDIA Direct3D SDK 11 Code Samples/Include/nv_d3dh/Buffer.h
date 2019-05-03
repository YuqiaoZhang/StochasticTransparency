//============================================================================
//============================================================================
#pragma once

//#include "Device.h"
//#include "../DXUT/Core/DXUT.h"

namespace d3d11
{

    class IBuffer
    {
    public:
        IBuffer(){}
        virtual ~IBuffer(){}

        virtual int  Size() const = 0;
        virtual void Resize(int a_size) = 0;
        virtual void CopyHostToDevice(ID3D11Device* a_pDevice) = 0;
        virtual void CopyDeviceToHost(ID3D11Device* a_pDevice) = 0;
        virtual void FreeCPUData() = 0;
        virtual void CreateNewBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice) = 0;
				virtual void CreateNewStructuredBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice) = 0;
        virtual void CreateNewDynamicBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice) = 0;
        
        virtual DXGI_FORMAT GetDXGIFormat() const = 0;
        virtual ID3D11Buffer* GetD3DBuffer() = 0;
        virtual const ID3D11Buffer* GetD3DBuffer() const = 0;

        virtual int GetBindFlags() const = 0;
        virtual int GetUsageFlags() const = 0;
    };

    template<class Elem>
    class Buffer : public IBuffer
    {
    public:

        Buffer();
        Buffer(int a_size);
        Buffer(Elem* a_data, int a_size);
        ~Buffer();
        virtual void SetFromData(Elem* a_data, int a_size);
        
        void Resize(int a_size);
        void CopyHostToDevice(ID3D11Device* a_pDevice);
        void CopyDeviceToHost(ID3D11Device* a_pDevice);
        void FreeCPUData();
        void CreateNewBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice);
				void CreateNewStructuredBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice);
        void CreateNewDynamicBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice);

        Elem& operator[](int a_index);
        const Elem& operator[](int a_index) const;

        ID3D11Buffer* GetD3DBuffer() 
        {
            if(m_pD3DBuffer==0) RUN_TIME_ERROR("A buffer have not yet a device copy");
            return m_pD3DBuffer;
        }

        const ID3D11Buffer* GetD3DBuffer() const 
        {
            if(m_pD3DBuffer==0) RUN_TIME_ERROR("A buffer have not yet a device copy");
            return m_pD3DBuffer;
        }

        Elem* GetPointer() 
        {
            if(m_cpuData==0) RUN_TIME_ERROR("A buffer have not yet a initialised");
            return m_cpuData;
        }

        const Elem* const GetPointer() const 
        {
            if(m_cpuData==0) RUN_TIME_ERROR("A buffer have not yet a initialised");
            return m_cpuData;
        }

        int Size() const { return m_size; }

        DXGI_FORMAT GetDXGIFormat() const {return DXGIFormat<Elem,32>();}

        int GetBindFlags() const  
        { 
            if(m_pD3DBuffer==NULL)
                return 0;

            D3D11_BUFFER_DESC desc;
            m_pD3DBuffer->GetDesc(&desc);
            return desc.BindFlags;
        }
        
        int GetUsageFlags() const 
        { 
            if(m_pD3DBuffer==NULL)
                return 0;

            D3D11_BUFFER_DESC desc;
            m_pD3DBuffer->GetDesc(&desc);
            return desc.Usage; 
        }

    protected:

        Elem* m_cpuData;
        int m_size;
        
        int m_oldSizeOnGPU;
        //int m_oldBindFlags;

        ID3D11Buffer* m_pD3DBuffer;

        enum 
        {
            D3D_GENERAL_BUFFER = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
            D3D_VERTEX_BUFFER = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE,
            D3D_INDEX_BUFFER = D3D11_BIND_INDEX_BUFFER | D3D11_BIND_SHADER_RESOURCE,
            D3D_CONSTANT_BUFFER = D3D11_BIND_CONSTANT_BUFFER | D3D11_BIND_SHADER_RESOURCE
        };

        virtual void CreateNewBufferOfKnownType(ID3D11Device* a_pDevice)
        {  
            if(GetBindFlags()!=0)
                CreateNewBuffer(m_size, GetBindFlags(), a_pDevice);
            else
                CreateNewBuffer(m_size, D3D_GENERAL_BUFFER, a_pDevice); 
        }

        void CopyDataToExistentBufferOnDevice(ID3D11Device* a_pDevice);
    };



    template<class Elem>
    class VertexBuffer : public Buffer<Elem>
    {
    public:
        VertexBuffer(): Buffer() {}
        VertexBuffer(Elem* a_data, int a_size) : Buffer(a_data, a_size){}
        ~VertexBuffer(){}

        void Attach(ID3D11DeviceContext* a_pContext);

        int GetDXGIFormat() 
        {
            RUN_TIME_ERROR("Do not call GetDXGIFormat for vertex buffer for a while");
            return 0;
        }

    protected:

        void CreateNewBufferOfKnownType(ID3D11Device* a_pDevice)
        {
           if(GetBindFlags()!=0)
            CreateNewBuffer(m_size, GetBindFlags(), a_pDevice);
           else
            CreateNewBuffer(m_size, D3D_VERTEX_BUFFER, a_pDevice); 
        }
    };


    class IndexBuffer : public Buffer<uint>
    {
    public:
        IndexBuffer(): Buffer() {}
        IndexBuffer(uint* a_data, int a_size) : Buffer(a_data, a_size){}
        ~IndexBuffer(){}

        void Attach(ID3D11DeviceContext* a_pContext);

    protected:

        void CreateNewBufferOfKnownType(ID3D11Device* a_pDevice)
        {
            if(GetBindFlags()!=0)
                CreateNewBuffer(m_size, GetBindFlags(), a_pDevice);
            else
                CreateNewBuffer(m_size, D3D_INDEX_BUFFER, a_pDevice); 
        }
    };

}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
d3d11::Buffer<Elem>::Buffer()
{
    m_cpuData = 0;
    m_pD3DBuffer = 0;
    m_size = 0;
    m_oldSizeOnGPU = 0;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
d3d11::Buffer<Elem>::Buffer(int a_size)
{
	m_cpuData = 0;
  m_pD3DBuffer = 0;
  m_size = 0;
  m_oldSizeOnGPU = 0;
	Resize(a_size);
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
d3d11::Buffer<Elem>::Buffer(Elem* a_data, int a_size)
{
	m_cpuData = 0;
  m_pD3DBuffer = 0;
  m_size = 0;
  m_oldSizeOnGPU = 0;
	Resize(a_size); 
  SetFromData(a_data, a_size);
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
d3d11::Buffer<Elem>::~Buffer()
{
    SAFE_RELEASE(m_pD3DBuffer);
    delete [] m_cpuData;
    m_cpuData = 0;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::Resize(int a_size)
{
    if(a_size==m_size)
        return;

    SAFE_RELEASE(m_pD3DBuffer);
    delete [] m_cpuData;

    m_cpuData = new Elem [a_size];
    m_size = a_size;
    m_oldSizeOnGPU = 0;
}


/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::SetFromData(Elem* a_data, int a_size)
{
    if(m_size != a_size)
        Resize(a_size);

    memcpy(m_cpuData, a_data, a_size*sizeof(Elem));
    m_size = a_size;
}


/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CreateNewBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice)
{
    Resize(a_size);

    SAFE_RELEASE(m_pD3DBuffer);

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
    bufferDesc.Usage            = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth        = m_size*sizeof(Elem);
    bufferDesc.BindFlags        = a_bindFlags;
    bufferDesc.CPUAccessFlags   = 0;
    bufferDesc.MiscFlags        = (a_bindFlags == D3D_GENERAL_BUFFER) ? D3D11_RESOURCE_MISC_BUFFER_STRUCTURED : 0;
    bufferDesc.StructureByteStride = (a_bindFlags == D3D_GENERAL_BUFFER) ? sizeof(Elem) : 0;

    D3D11_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = m_cpuData;
    InitData.SysMemPitch = 0;
    InitData.SysMemSlicePitch = 0;

    if(FAILED(a_pDevice->CreateBuffer(&bufferDesc, &InitData, &m_pD3DBuffer)))
        RUN_TIME_ERROR("CreateBuffer failed.");

    m_oldSizeOnGPU = m_size;
    //m_oldBindFlags = a_bindFlags;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CreateNewStructuredBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice)
{
    Resize(a_size);

    SAFE_RELEASE(m_pD3DBuffer);

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
    bufferDesc.Usage            = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth        = m_size*sizeof(Elem);
    bufferDesc.BindFlags        = a_bindFlags;
    bufferDesc.CPUAccessFlags   = 0;
    bufferDesc.MiscFlags        = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufferDesc.StructureByteStride = (a_bindFlags == D3D_GENERAL_BUFFER) ? sizeof(Elem) : 0;

    D3D11_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = m_cpuData;
    InitData.SysMemPitch = 0;
    InitData.SysMemSlicePitch = 0;

    if(FAILED(a_pDevice->CreateBuffer(&bufferDesc, &InitData, &m_pD3DBuffer)))
        RUN_TIME_ERROR("CreateBuffer failed.");

    m_oldSizeOnGPU = m_size;
    //m_oldBindFlags = a_bindFlags;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CreateNewDynamicBuffer(int a_size, int a_bindFlags, ID3D11Device* a_pDevice)
{
    Resize(a_size);

    SAFE_RELEASE(m_pD3DBuffer);

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
    bufferDesc.Usage            = D3D11_USAGE_DYNAMIC;
    bufferDesc.ByteWidth        = m_size*sizeof(Elem);
    bufferDesc.BindFlags        = a_bindFlags;
    bufferDesc.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    bufferDesc.MiscFlags        = (a_bindFlags == D3D_GENERAL_BUFFER) ? D3D11_RESOURCE_MISC_BUFFER_STRUCTURED : 0;
    bufferDesc.StructureByteStride = (a_bindFlags == D3D_GENERAL_BUFFER) ? sizeof(Elem) : 0;

    D3D11_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = m_cpuData;
    InitData.SysMemPitch = 0;
    InitData.SysMemSlicePitch = 0;

    if(FAILED(a_pDevice->CreateBuffer(&bufferDesc, &InitData, &m_pD3DBuffer)))
        RUN_TIME_ERROR("CreateBuffer failed.");

    m_oldSizeOnGPU = m_size;
    //m_oldBindFlags = a_bindFlags;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CopyHostToDevice(ID3D11Device* a_pDevice)
{
    if(m_pD3DBuffer == 0 || m_oldSizeOnGPU != m_size)
        CreateNewBufferOfKnownType(a_pDevice);
    else
        CopyDataToExistentBufferOnDevice(a_pDevice);

    m_oldSizeOnGPU = m_size;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CopyDeviceToHost(ID3D11Device* a_pDevice)
{
    if(m_pD3DBuffer == 0 || m_oldSizeOnGPU != m_size)
        RUN_TIME_ERROR("A buffer on 'Device' was not created");

    ID3D11Buffer* tempBuffer;

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
    bufferDesc.Usage            = D3D11_USAGE_STAGING;
    bufferDesc.ByteWidth        = m_size*sizeof(Elem);
    bufferDesc.BindFlags        = 0; // ??????
    bufferDesc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

    DX_SAFE_CALL(a_pDevice->CreateBuffer(&bufferDesc, 0, &tempBuffer));

    ID3D11DeviceContext* pd3dImmediateContext = GetCtx(a_pDevice);
    pd3dImmediateContext->CopyResource(tempBuffer, m_pD3DBuffer);

    D3D11_MAPPED_SUBRESOURCE MappedResource;
    DX_SAFE_CALL(pd3dImmediateContext->Map(tempBuffer, 0, D3D11_MAP_READ, 0, &MappedResource ));

    memcpy(m_cpuData, MappedResource.pData, m_size*sizeof(Elem) );

    pd3dImmediateContext->Unmap(tempBuffer, 0 );

    SAFE_RELEASE(tempBuffer);
    SAFE_RELEASE(pd3dImmediateContext);
}


/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::CopyDataToExistentBufferOnDevice(ID3D11Device* a_pDevice)
{
    ID3D11Buffer* tempBuffer;
    ID3D11DeviceContext* pd3dImmediateContext = 0;
    
    a_pDevice->GetImmediateContext(&pd3dImmediateContext);
    if(pd3dImmediateContext==0)
        RUN_TIME_ERROR("GetImmediateContext failed");

    D3D11_BUFFER_DESC thisBufferDesc;
    m_pD3DBuffer->GetDesc(&thisBufferDesc);

    if (thisBufferDesc.Usage == D3D11_USAGE_DEFAULT)
    {
        pd3dImmediateContext->UpdateSubresource(m_pD3DBuffer,0,0,m_cpuData,0,0);
    }
    else if (thisBufferDesc.Usage == D3D11_USAGE_DYNAMIC)
    {
        RUN_TIME_ERROR("untested copy for dynamic buffers");
        D3D11_MAPPED_SUBRESOURCE res;
        
        pd3dImmediateContext->Map(m_pD3DBuffer,0,D3D11_MAP_WRITE,0,&res);
        
        Elem* arr = (Elem*)res.pData;
        for(int i=0;i<m_size/4;i+=4)
        {
            arr[i+0] = m_cpuData[i+0];
            arr[i+1] = m_cpuData[i+1];
            arr[i+2] = m_cpuData[i+2];
            arr[i+3] = m_cpuData[i+3];
        }

        pd3dImmediateContext->Unmap(m_pD3DBuffer, 0);    
    }
    else
    {
        D3D11_BUFFER_DESC bufferDesc;
        ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
        bufferDesc.Usage            = D3D11_USAGE_DEFAULT;
        bufferDesc.ByteWidth        = m_size*sizeof(Elem);

        D3D11_SUBRESOURCE_DATA InitData;
        InitData.pSysMem = m_cpuData;
        InitData.SysMemPitch = 0;
        InitData.SysMemSlicePitch = 0;

        DX_SAFE_CALL(a_pDevice->CreateBuffer(&bufferDesc, &InitData, &tempBuffer));

        pd3dImmediateContext->CopyResource(m_pD3DBuffer, tempBuffer);

        SAFE_RELEASE(tempBuffer);
    }
    
    SAFE_RELEASE(pd3dImmediateContext);
}


/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::Buffer<Elem>::FreeCPUData()
{
    delete [] m_cpuData;
    m_cpuData = 0;
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
Elem& d3d11::Buffer<Elem>::operator[](int a_index)
{
    ASSERT(m_cpuData!=0);
    ASSERT(a_index < m_size);
    return m_cpuData[a_index];
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
const Elem& d3d11::Buffer<Elem>::operator[](int a_index) const
{
    ASSERT(m_cpuData!=0);
    ASSERT(a_index < m_size);
    return m_cpuData[a_index];
}

/////////////////////////////////////////////////////////////////////////////////
////
template<class Elem>
void d3d11::VertexBuffer<Elem>::Attach(ID3D11DeviceContext* pContext)
{
    if(pContext == 0)
        throw std::runtime_error("VertexBuffer::Attach(): zero intermediate context");

    uint offset = 0;
    uint stride = sizeof(Elem);

    pContext->IASetVertexBuffers(0, 1, &m_pD3DBuffer, &stride, &offset);
}


