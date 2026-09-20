# GrainCounter

Aplicativo Windows para contagem de grãos/unidades em superfície plana e uniforme.

## Recursos

- Seleção de webcam integrada ou câmera USB.
- Contagem em tempo real.
- Destaque rosa dos objetos contabilizados.
- Numeração individual.
- Separação de objetos encostados por transformada de distância + watershed.
- Estabilização temporal.
- Filtros de área.
- **Calibração automática** para a câmera, superfície e material.
- Build automático do `.exe` por GitHub Actions.

## Você NÃO precisa instalar Python

O GitHub Actions cria o executável em um computador Windows hospedado pelo GitHub. O PyInstaller empacota o interpretador Python e as dependências no aplicativo, portanto o computador de uso final não precisa ter Python instalado.

## Gerar o EXE pelo GitHub

1. Crie um repositório no GitHub.
2. Envie todos os arquivos deste projeto.
3. Vá em **Actions**.
4. Execute **Build Windows EXE** manualmente.
5. Abra o workflow concluído.
6. Baixe o artefato **GrainCounter-Windows**.

Também é possível criar uma tag como `v1.0.0`; nesse caso o workflow cria automaticamente uma Release com `GrainCounter.exe`.

## Calibração

Com a câmera iniciada, coloque uma quantidade representativa de objetos na superfície, incluindo alguns que estejam próximos/encostados. Clique em **Calibrar automaticamente**.

A calibração estima:
- sensibilidade de segmentação;
- área mínima;
- área máxima.

Depois você pode ajustar os parâmetros manualmente.

## Evolução para IA

A versão atual prioriza uma superfície uniforme e usa visão computacional clássica, o que é apropriado quando o fundo é controlado. Para máxima robustez com formas, cores e sobreposições muito variadas, a próxima evolução é um modelo de segmentação de instâncias treinado especificamente com imagens reais do material.
