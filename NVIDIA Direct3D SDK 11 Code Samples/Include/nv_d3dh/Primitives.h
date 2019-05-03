#pragma once

#include "D3DHelpers.h"
#include "Buffer.h"
#include "Texture.h"

namespace d3d11
{


class Quad
{

public:

    Quad(ID3D11Device* a_pDevice, ID3DX11EffectTechnique* a_pTechniqueDrawQuad);
    virtual ~Quad();

    virtual void Draw();
    virtual void DrawCall();

protected:
    
    struct VPosTex
    {
        D3DXVECTOR3 pos;
        D3DXVECTOR2 t;
    };

    ID3D11Device* m_pDevice;
    ID3D11DeviceContext* m_pContext;
    ID3DX11EffectTechnique* m_pTechnique;

    d3d11::VertexBuffer<VPosTex> m_quadVB; 
    d3d11::IndexBuffer m_quadIB;
    ID3D11InputLayout* m_pQuadVertexlayout;
};

class FullScreenQuad: public Quad
{
public:
    FullScreenQuad(ID3D11Device* a_pDevice,ID3DX11EffectTechnique* a_pTechniqueDrawQuad): Quad(a_pDevice,a_pTechniqueDrawQuad) {}
    ~FullScreenQuad(){}
};



}

